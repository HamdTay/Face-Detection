
import org.opencv.objdetect.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.*;
import org.opencv.core.*;
import java.util.List;
import java.util.Scanner;
/*
 * Load Haar Classifier
 * */


public class FaceDetect {
	  

	
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String classifierfile = "HAARClassifier\\haarcascade_frontalface_alt.xml";
		CascadeClassifier faceCascade = new CascadeClassifier();
		//Load classifier
		if(!faceCascade.load(classifierfile))
			System.err.println("Cannot load classifier");
		
		//TakePicture();
		FaceDetect face = new FaceDetect();
		Mat image = face.TakePicture();
		if(image == null)
			System.err.println("The image is empty!!");
		
		//process image
		Mat imageGrey = new Mat();
		Imgproc.cvtColor(image, imageGrey, Imgproc.COLOR_RGB2GRAY);
		Imgproc.equalizeHist(imageGrey, imageGrey);
		
		//detectFace
		face.DetectFace(image, imageGrey, faceCascade);
	}
	
	//press enter to take picture
	//
	
	public void ShowImage(Mat M) {
		if(M == null) {
			System.err.println("Image is empty--imshow fun--");
		}
		HighGui.imshow("--", M);
		HighGui.waitKey();
	}
	public Mat TakePicture() {
		
		VideoCapture V = new VideoCapture(0, Videoio.CAP_DSHOW);
		Mat M = new Mat(),image = new Mat();
		if (V.isOpened()) {
			while (true) {
				try {
					V.read(image);
					if(image.empty()) {
						System.out.println("Error: empty image");
						return null;
					}
					Thread.sleep(5);
					HighGui.imshow("Webcam", image);
					//Press enter to take a picture
					int index = HighGui.waitKey(30);
					//if enter is clicked take a picture, and write it to workspace
					if(index == 10) {
						Imgcodecs.imwrite("Photo personnel.jpg", image);
						V.release();
						return image;
					}
					//if ESC is clicked exit without taking a photo
					else if(index == 27) {
						System.out.println("Photo not taken");
						V.release();
						return null;
					}
					
				}catch(InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		V.release();
		return image;

	}
	
	
	public void DetectFace(Mat image, Mat greyimg, CascadeClassifier faceCascade) {
		MatOfRect faces = new MatOfRect();
		faceCascade.detectMultiScale(greyimg, faces);
		//convertir matrices des rectangle a une liste
		List<Rect> listOfFaces = faces.toList();
		Mat OnlyFace = new Mat();
		for(Rect face : listOfFaces) {
			//take only the face
			OnlyFace = image.submat(face);
			//draw a rectangle on the face
			Imgproc.rectangle(image, face , new Scalar(51,255,255), Imgproc.CV_SHAPE_RECT);
		}
		if(image != null) 
			HighGui.imshow("Detect Face", image);
		
		if(OnlyFace != null) {
			HighGui.imshow("Only face", OnlyFace);
			MatOfInt M = new MatOfInt();
			M.alloc(Imgcodecs.IMWRITE_PXM_BINARY);
			//still have to resize the photo to mach the dataset
			Imgproc.cvtColor(OnlyFace, OnlyFace, Imgproc.COLOR_BGR2GRAY);
			Imgcodecs.imwrite("Only Face.pgm", OnlyFace, M);
		}
		HighGui.waitKey(0);
	}
	
	

}