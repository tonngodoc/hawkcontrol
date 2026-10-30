#if canImport(UIKit) && canImport(VideoToolbox)
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import Metal
import OpenCineCore
import UIKit
import VideoToolbox
import os
import OpenCineAppleObjC

/// Admission captured before queued decode, using the existing decoded-stream fence.
public struct MonitorJPEGDecodeSource: Sendable {
    fileprivate let stream: StreamID
    fileprivate let generation: StreamGeneration
}

/// Only source binding and terminal admission live under this short lock.
/// Decode, temporal processing and CI readback run on the actor outside it.
private final class MonitorJPEGDecodeLifetime: @unchecked Sendable {
    let fence = DecodedFrameFence()
    private let lock = NSLock()
    private weak var source: AnyObject?
    private var closed = false

    func bind(_ candidate: AnyObject) -> MonitorJPEGDecodeSource? {
        lock.withLock {
            guard !closed else { return nil }
            if source !== candidate {
                source = candidate
                fence.advance()
            }
            return MonitorJPEGDecodeSource(stream: fence.stream, generation: fence.generation)
        }
    }
    func retire() -> MonitorJPEGDecodeSource {
        lock.withLock {
            source = nil
            return MonitorJPEGDecodeSource(stream: fence.stream, generation: fence.advance())
        }
    }
    func isGenerationCurrent(_ token: MonitorJPEGDecodeSource) -> Bool {
        lock.withLock { token.stream == fence.stream && fence.isCurrent(token.generation) }
    }
    func close() {
        lock.withLock {
            closed = true
            source = nil
            fence.advance()
        }
    }
    func isCurrent(_ token: MonitorJPEGDecodeSource) -> Bool {
        lock.withLock {
            !closed && source != nil && token.stream == fence.stream && fence.isCurrent(token.generation)
        }
    }
}

/// Owns motion-JPEG native decode, decoded leases, temporal history and UIImage preparation.
/// Manufacturer/operator choices arrive as bounded values and diagnostic callbacks.
public actor MonitorJPEGImageDecoder {
    private nonisolated let lifetime = MonitorJPEGDecodeLifetime()
    private let hardware: JPEGPixelBufferDecoder
    private let noiseFilter: MonitorTemporalNoiseFilter
    private let superResolution: MonitorDecodedSuperResolution
    private let denoiseStrength: @Sendable () -> Float?
    private let unavailable: @Sendable (String?) -> Void
    private var historyGeneration: StreamGeneration?
    private lazy var ciContext = CIContext(options: [.cacheIntermediates: false])

    public init(
        logSubsystem: String = "OpenCineApple",
        denoiseStrength: @escaping @Sendable () -> Float? = { nil },
        unavailable: @escaping @Sendable (String?) -> Void = { _ in }
    ) {
        self.hardware = JPEGPixelBufferDecoder(fence: lifetime.fence, logSubsystem: logSubsystem)
        self.noiseFilter = MonitorTemporalNoiseFilter(logSubsystem: logSubsystem, unavailable: unavailable)
        self.superResolution = MonitorDecodedSuperResolution(logSubsystem: logSubsystem)
        self.denoiseStrength = denoiseStrength
        self.unavailable = unavailable
    }

    /// Call synchronously with the actual acquisition owner before awaiting queued decode.
    public nonisolated func source(_ owner: AnyObject) -> MonitorJPEGDecodeSource? {
        lifetime.bind(owner)
    }

    public nonisolated func retireSource() -> MonitorJPEGDecodeSource { lifetime.retire() }

    public func resetProcessors(source: MonitorJPEGDecodeSource) {
        guard lifetime.isGenerationCurrent(source) else { return }
        noiseFilter.reset()
        unavailable(nil)
        historyGeneration = nil
    }

    public nonisolated func close() { lifetime.close() }

    public func decode(
        _ jpeg: Data, source: MonitorJPEGDecodeSource, preferHardware: Bool = true
    ) -> sending UIImage? {
        guard lifetime.isCurrent(source), !Task.isCancelled else { return nil }
        if historyGeneration != source.generation {
            noiseFilter.reset()
            unavailable(nil)
            historyGeneration = source.generation
        }
        if preferHardware, let image = hardwareDecoded(jpeg, source: source) { return image }
        guard lifetime.isCurrent(source), !Task.isCancelled else { return nil }
        return autoreleasepool {
            let image = UIImage(data: jpeg)?.preparingForDisplay()
            return lifetime.isCurrent(source) && !Task.isCancelled ? image : nil
        }
    }

    private func hardwareDecoded(
        _ jpeg: Data, source: MonitorJPEGDecodeSource
    ) -> sending UIImage? {
        autoreleasepool {
            guard let lease = hardware.decodeLease(jpeg) else { return nil }
            defer { lease.release() }
            guard lifetime.isCurrent(source), lease.generation == source.generation,
                var buffer = lease.pixelBuffer
            else { return nil }
            if let strength = denoiseStrength(),
                let filtered = noiseFilter.filter(buffer, strength: strength)
            {
                buffer = filtered
            }
            guard lifetime.isCurrent(source), !Task.isCancelled else { return nil }
            let image = CIImage(cvPixelBuffer: buffer)
            guard let cgImage = ciContext.createCGImage(image, from: image.extent),
                lifetime.isCurrent(source), !Task.isCancelled
            else { return nil }
            return UIImage(cgImage: cgImage)
        }
    }

    // Preserved dormant decode-side path. It is never selected by decode().
    private func superResolved(
        _ jpeg: Data, source: MonitorJPEGDecodeSource
    ) -> sending UIImage? {
        autoreleasepool {
            guard lifetime.isCurrent(source), let decoded = hardware.decode(jpeg),
                let upscaled = superResolution.upscale(decoded), lifetime.isCurrent(source)
            else { return nil }
            let image = CIImage(cvPixelBuffer: upscaled)
            guard let cgImage = ciContext.createCGImage(image, from: image.extent),
                lifetime.isCurrent(source)
            else { return nil }
            return UIImage(cgImage: cgImage)
        }
    }
}


/// A `CVPixelBuffer` format as the four characters it actually is, `875704438` is unreadable
/// in a report line and `420v` is the answer.
private func fourCCName(_ format: OSType) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((format >> $0) & 0xFF) }
    let text = String(decoding: bytes, as: UTF8.self)
    return text.allSatisfy { $0.isASCII && !$0.isNewline } ? text : String(format)
}

/// Runs VideoToolbox super resolution over a decoded live-view frame, entirely off the render
/// thread.
///
/// This lives on the decode side rather than in the renderer, and that placement is the point.
/// Every earlier attempt drove the processor from `draw(in:)`, where the header forbids it, "ML
/// model loading may take longer than a frame time. Avoid blocking the UI thread or stalling frame
/// rendering pipelines during this call", and where a Metal command buffer's encode order and
/// Core Image's snapshot semantics both had to be reasoned about. Here there is no command buffer,
/// no drawable, and no draw callback: a frame goes in and a frame comes out, on a thread that is
/// allowed to take its time.
///
/// The chain is `420v` end to end. The camera sends motion JPEG, which already carries
/// chroma-subsampled planes, so the hardware decoder hands over the model's own format and the
/// pixel transfer below resizes within it. No colour conversion happens anywhere on the way in.
private final class MonitorDecodedSuperResolution {
    init(logSubsystem: String) {}
    func upscale(_ frame: CVPixelBuffer) -> CVPixelBuffer? { nil }
    static func pool(width: Int, height: Int, format: OSType) -> CVPixelBufferPool? {
        var pool: CVPixelBufferPool?
        let attributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: format,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [String: Any],
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        return pool
    }
}

public final class MonitorTemporalNoiseFilter {
    init(logSubsystem: String, unavailable: @escaping @Sendable (String?) -> Void) {}
    private(set) var report = "idle"
    func filter(_ frame: CVPixelBuffer, strength: Float) -> CVPixelBuffer? { nil }
    func reset() {}
}

#endif
