package com.example.my_opencv;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.content.res.AssetManager;

import android.graphics.Camera;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerKCF;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.*;
import android.os.Bundle;


public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    // Initialize OpenCV manager.
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.setCameraIndex(1);
                   // mOpenCvCameraView.setRotation(180);
                    mOpenCvCameraView.enableView();


                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
    }


    //when app starts
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        // Set up camera listener.
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        //mytext = (TextView) findViewById(R.id.myText);
    }

    //Load a dnn network.
    public void onCameraViewStarted(int width, int height) {
        String proto = getPath("MobileNetSSD_deploy.prototxt", this);
        String weights = getPath("MobileNetSSD_deploy.caffemodel", this);
        net = Dnn.readNetFromCaffe(proto, weights);
        Log.i(TAG, "Network loaded successfully");

        //create kcf tracker and init variables
        tracker = TrackerKCF.create();
        find = true;
        counter = 0;

    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        // Get a new frame and convert colors
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        //reseize
       // Imgproc.resize(frame,frame, new Size(1280,720));

        //label for text.
        String label ="";

        //find person
        if (find) {

            frame = identify(frame);
        }

        //track person
        else {

            //start tick
            long ticker = Core.getTickCount();

            //look for person
            boolean ok = tracker.update(frame, bbox);

            //get difference
            long fps = (long) Core.getTickFrequency() / (Core.getTickCount() - ticker);

            //update tracker status
            if (ok) {
                label = "Status: Tracking ";

            } else {
                label = "Status: Lost Tracking ";

                //if more than 50 frames, look for a new person to track
                counter++;
                if(counter >= 50){
                    find = true;
                    label = "Reseting Tracker ";
                    counter = 0;
                    tracker = TrackerKCF.create();
                }

            }

            //show rect of person found
            Imgproc.rectangle(frame, new Point(bbox.x, bbox.y), new Point((bbox.width + bbox.x), (bbox.height + bbox.y)), new Scalar(0, 0, 255), 5);

            // show tracking status label
            Imgproc.putText(frame, label + ", fps: " + fps, new Point(30, 90),
                    Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255,0,0), 5);
        }

        return frame;

        // Draw rectangle around detected object.
//                    rect = Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
        //                          new Scalar(0, 255, 0));
//                    String label = classNames[classId] + ": " + confidence;
//                    int[] baseLine = new int[1];
//                    Size labelSize = Imgproc.getTextSize(label, Core.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
//
//                    // Draw background for label.
//                    Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
//                            new Point(left + labelSize.width, top + baseLine[0]),
//                            new Scalar(255, 255, 255));
//
//                    // Write class name and confidence.
//                    Imgproc.putText(frame, label, new Point(left, top),
//                            Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));

    }

    public void onCameraViewStopped() {
    }

    //find person
    public Mat identify(Mat frame) {

        //dnn setup
        final int IN_WIDTH = 300;
        final int IN_HEIGHT = 300;
        final float WH_RATIO = (float) IN_WIDTH / IN_HEIGHT;
        final double IN_SCALE_FACTOR = 0.007843;
        final double MEAN_VAL = 127.5;
        final double THRESHOLD = 0.7;

        // Forward image through network.
        Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
                new Size(IN_WIDTH, IN_HEIGHT),
                new Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false);

        net.setInput(blob);
        Mat detections = net.forward();

        int cols = frame.cols();
        int rows = frame.rows();

        detections = detections.reshape(1, (int) detections.total() / 7);

        for (int i = 0; i < detections.rows(); ++i) {

            double confidence = detections.get(i, 2)[0];
            int classId = (int) detections.get(i, 1)[0];

            //proceed with confidence and person detected
            if (confidence > THRESHOLD && classId == 15) {

                //get bbox
                int left = (int) (detections.get(i, 3)[0] * cols);
                int top = (int) (detections.get(i, 4)[0] * rows);
                int right = (int) (detections.get(i, 5)[0] * cols);
                int bottom = (int) (detections.get(i, 6)[0] * rows);

                int factor = 20;

                //create rect for init
                bbox = new Rect2d(new Point(left + factor , top + factor), new Point(right - factor, bottom - factor));

                //init tracker
               tracker.init(frame, bbox);

                //show rect
                Imgproc.rectangle(frame, new Point(left + factor, top + factor), new Point(right - factor, bottom - factor), new Scalar(0, 255, 0), 5);

                find = false;
            }
        }

        return frame;
    }


    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {

        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;

        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);

            os.write(data);
            os.close();

            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();

        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }

        return "";
    }

    private static final String TAG = "OpenCV/Sample/MobileNet";
    private static final String[] classNames = {"background", "plane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "dinningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "monitor"};
    private Net net;
    private CameraBridgeViewBase mOpenCvCameraView;
    private TrackerKCF tracker;
    private boolean find;
    private Rect2d bbox;
    Camera camera;
    int counter;

}