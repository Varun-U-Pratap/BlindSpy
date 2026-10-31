// Video helper for the demo recording (macOS only; AVFoundation, no ffmpeg needed).
//
//   encode <frames dir> <fps> <out.mp4>   PNG frames (sorted by name) -> H.264 MP4
//   grab <video.mp4> <out dir> <seconds...>   one PNG still per time, to check a video
//
// Run: swiftc -O scripts/encode_video.swift -o <scratch>/encode_video, then the binary.

import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func loadImage(_ url: URL) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail("cannot read \(url.path)") }
    return img
}

func encode(dir: String, fps: Int32, out: String) {
    let fm = FileManager.default
    let names = (try? fm.contentsOfDirectory(atPath: dir))?.filter { $0.hasSuffix(".png") }.sorted() ?? []
    if names.isEmpty { fail("no PNG frames in \(dir)") }
    let first = loadImage(URL(fileURLWithPath: dir).appendingPathComponent(names[0]))
    let (w, h) = (first.width, first.height)
    let outURL = URL(fileURLWithPath: out)
    try? fm.removeItem(at: outURL)
    guard let writer = try? AVAssetWriter(outputURL: outURL, fileType: .mp4) else { fail("cannot create \(out)") }
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
        AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 1_500_000,
                                          AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
    ])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
    ])
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)
    for (i, name) in names.enumerated() {
        let img = loadImage(URL(fileURLWithPath: dir).appendingPathComponent(name))
        if img.width != w || img.height != h { fail("\(name) is \(img.width)x\(img.height), expected \(w)x\(h)") }
        while !input.isReadyForMoreMediaData { usleep(1000) }
        var buf: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buf)
        guard let pb = buf else { fail("no pixel buffer") }
        CVPixelBufferLockBaseAddress(pb, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(pb, [])
        if !adaptor.append(pb, withPresentationTime: CMTime(value: Int64(i), timescale: fps)) {
            fail("append failed at frame \(i): \(String(describing: writer.error))")
        }
    }
    input.markAsFinished()
    let done = DispatchSemaphore(value: 0)
    writer.finishWriting { done.signal() }
    done.wait()
    if writer.status != .completed { fail("writing failed: \(String(describing: writer.error))") }
    print("wrote \(out): \(names.count) frames, \(w)x\(h), \(fps) fps")
}

func grab(video: String, outDir: String, times: [Double]) {
    let gen = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: video)))
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .zero
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    for t in times {
        guard let img = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil)
        else { fail("no frame at \(t) s") }
        let url = URL(fileURLWithPath: outDir).appendingPathComponent(String(format: "t%05.1f.png", t))
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, img, nil)
        CGImageDestinationFinalize(dest)
        print("wrote \(url.path)")
    }
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "encode" where args.count == 5:
    encode(dir: args[2], fps: Int32(args[3]) ?? 10, out: args[4])
case "grab" where args.count >= 5:
    grab(video: args[2], outDir: args[3], times: args[4...].compactMap(Double.init))
default:
    fail("usage: encode <frames dir> <fps> <out.mp4> | grab <video.mp4> <out dir> <seconds...>")
}
