//
//  HeatmapVideoController.swift
//  OpenPose
//
//  Created by ben on 2019/7/29.
//  Copyright © 2019 ben. All rights reserved.
//

import UIKit
import Vision
import AVFoundation

class HeatmapVideoController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    
    let serialQueue = DispatchQueue.init(label: "", qos: .default, attributes: [.concurrent], autoreleaseFrequency: .inherit, target: nil)
    
    let model = pose_368()
    
    let ImageWidth = 368
    let ImageHeight = 368
    
    var scaleX: CGFloat = 1.0
    var scaleY: CGFloat = 1.0
    var magrin: CGFloat = 0.0
    
    let session = AVCaptureSession()
    let deviceInput = DeviceInput()
    let output = AVCaptureVideoDataOutput()
    
    typealias FilterCompletion = ((UIImage?, InferenceError) -> ())
    
    var lines = [CAShapeLayer]()
    
    var isProgressing = false
    
    override func viewDidLoad() {

        super.viewDidLoad()
        
        // set input, output and session
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video-stream"))
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String : kCVPixelFormatType_32BGRA]
        
        // session.sessionPreset = AVCaptureSession.Preset.hd1280x720
        session.sessionPreset = AVCaptureSession.Preset.vga640x480
        // session.sessionPreset = AVCaptureSession.Preset.cif352x288
        session.addInput(deviceInput.backWildAngleCamera!)
        session.addOutput(output)
        
        
        let connection = output.connection(with: .video)
        connection?.videoOrientation = .portrait
        
        session.startRunning()

    }

    // MARK: - fuctions relative to CoreML
    
    func process(input: UIImage, complition: @escaping FilterCompletion) {
        print("process start ")
        // resize the image
        guard let inputImage = input.resize(to: CGSize(width: 368, height: 368)) else {
            complition(nil, InferenceError.resizeError)
            return
        }
        
        // convert the image
        guard let cvBufferInput = inputImage.pixelBuffer() else{
            complition(nil, InferenceError.pixelBufferError)
            return
        }
        
        // feed the image to the neural network
        guard let output = try? self.model.prediction(input_image: cvBufferInput) else {
            complition(nil, InferenceError.predictionError)
            return
        }
        
        //
        // process the heatnap
        // output.heat_map_2 => (1,1,n, w, h)
        //
        let keypoint_number = output.heat_map_2.shape[2].int32Value
        let heatmap_w = output.heat_map_2.shape[3].int32Value
        let heatmap_h = output.heat_map_2.shape[4].int32Value
        
//        print("keypoint_number = \(keypoint_number), heatmap_w = \(heatmap_w), heatmap_h = \(heatmap_h)")
        
        var tensorShape:[Int32] = [heatmap_w, heatmap_h, keypoint_number]
        let convertedHeatMap = OpenCVWrapper.visualizeHeatmap(
                                                output.heat_map_2,
                                                heatmapShape: &tensorShape,
                                                inputImage: input)
        
        
        
        
        
        serialQueue.async {
            if self.isProgressing {
                return
            }
            let startTime = CFAbsoluteTimeGetCurrent()
            self.isProgressing = true
            let pafarr = self.getArr(mlarr: output.paf_2)
            let heatmaparr = self.getArr(mlarr: output.heat_map_2)
            let com = PoseEstimator(368,368)
            let humans = com.estimate(heatmap: heatmaparr.asArrayOfDouble, paf: pafarr.asArrayOfDouble)
            print("got humans = \(humans.count)")
            DispatchQueue.main.async {
                self.lines.map { layer in
                    layer.removeFromSuperlayer()
                }
                self.lines.removeAll()
                self.drawLine(humans: humans)
                self.isProgressing = false
                let timePrigress = CFAbsoluteTimeGetCurrent() - startTime
                print("progres for \(timePrigress) seconds")
            }
        }
        
        
//        let pafarr = self.getArr(mlarr: output.paf_2)
//        let heatmaparr = self.getArr(mlarr: output.heat_map_2)
//        let com = PoseEstimator(368,368)
//        let humans = com.estimate(heatmap: heatmaparr.asArrayOfDouble, paf: pafarr.asArrayOfDouble)
//        print("got humans = \(humans.count)")
//        self.lines.map { layer in
//            layer.removeFromSuperlayer()
//        }
//        self.lines.removeAll()
//        self.drawLine(humans: humans)
        
        

//
//        self.imageView.image = convertedHeatMap
        print("process end ")
    } // process
    
    
    func getArr(mlarr : MLMultiArray) -> Array<Float> {
        let length = mlarr.count
        let floatPtr =  mlarr.dataPointer.bindMemory(to: Float32.self, capacity: length)
        let floatBuffer = UnsafeBufferPointer(start: floatPtr, count: length)
        let arr = Array(floatBuffer)
        return arr
    }
    
    func drawLine(humans: [Human]){
        
        var keypoint = [Int32]()
        var pos = [CGPoint]()
        for human in humans {
            var centers = [Int: CGPoint]()
            for i in 0...CocoPart.Background.rawValue {
                if human.bodyParts.keys.index(of: i) == nil {
                    continue
                }
                
                let bodyPart = human.bodyParts[i]!
                if bodyPart.partIdx > 13 || bodyPart.partIdx < 8 {
                    //continue
                }
//                print("bodyPart: ",bodyPart.x, bodyPart.y)
                centers[i] = CGPoint(x: bodyPart.x, y: bodyPart.y)
                centers[i] = CGPoint(x: Int(bodyPart.x * CGFloat(ImageWidth) * scaleX  + 0.5), y: (Int(bodyPart.y * CGFloat(ImageHeight) * scaleY +  magrin + 0.5)))
                
            }
            
            for (pairOrder, (pair1,pair2)) in CocoPairsRender.enumerated() {
                
                if human.bodyParts.keys.index(of: pair1) == nil || human.bodyParts.keys.index(of: pair2) == nil {
                    continue
                }
                if centers.index(forKey: pair1) != nil && centers.index(forKey: pair2) != nil{
                    keypoint.append(Int32(pairOrder))
                    pos.append(centers[pair1]!)
                    pos.append(centers[pair2]!)
                    addLine(fromPoint: centers[pair1]!, toPoint: centers[pair2]!, color: CocoColors[pairOrder])
                }
            }
        }
    }
    
    func addLine(fromPoint start: CGPoint, toPoint end:CGPoint, color: UIColor) {
        let line = CAShapeLayer()
        let linePath = UIBezierPath()
        
        linePath.move(to: start)
        linePath.addLine(to: end)
        line.path = linePath.cgPath
        line.strokeColor = color.cgColor
        line.lineWidth = 3
        line.lineJoin = CAShapeLayerLineJoin.round
        
        lines.append(line)
        self.view.layer.addSublayer(line)
    }
    
}

// MARK: AVCaptureVideoDataOutputSampleBufferDelegate

extension HeatmapVideoController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Fail to get pixel buffer")
            return
        }

        if let captureImage = buffer.toUIImage() {
            DispatchQueue.main.sync {
                self.imageView.image = captureImage
                let w = imageView.frame.width
                let h = w * ((imageView.image?.size.height)!/(imageView.image?.size.width)!)
                scaleX = w/368
                scaleY = h/368
                magrin = (imageView.frame.height - h)/2
                self.process(input: self.imageView.image!) { image, error in
                    print(error)
                }
            }
        }
    }
    
}

extension Array where Element == Float32 {
    public var asArrayOfDouble: [Double] {
        return self.map { return Double($0) } // compiler error
    }
}
