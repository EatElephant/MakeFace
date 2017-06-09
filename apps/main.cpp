/*
 * 4dface: Real-time 3D face tracking and reconstruction from 2D video.
 *
 * File: apps/4dface.cpp
 *
 * Copyright 2015, 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "helpers.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/render.hpp"
#include "eos/render/texture_extraction.hpp"

#include "rcr/model.hpp"
#include "cereal/cereal.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"

#include "Eigen/Dense"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <exception>
#include <vector>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Rect;
using std::cout;
using std::endl;
using std::vector;
using std::string;

void draw_axes_topright(float r_x, float r_y, float r_z, cv::Mat image);

/**
 * This app demonstrates facial landmark tracking, estimation of the 3D pose
 * and fitting of the shape model of a 3D Morphable Model from a video stream,
 * and merging of the face texture.
 */
int main(int argc, char *argv[])
{
	fs::path modelfile, facedetector, landmarkdetector, mappingsfile, contourfile, edgetopologyfile, blendshapesfile;
	string input_prefix, input_ext;
	int input_frames;
	bool useVideo = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("morphablemodel,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				"a Morphable Model stored as cereal BinaryArchive")
			("facedetector,f", po::value<fs::path>(&facedetector)->required()->default_value("../share/haarcascade_frontalface_alt2.xml"),
				"full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
			("landmarkdetector,l", po::value<fs::path>(&landmarkdetector)->required()->default_value("../share/face_landmarks_model_rcr_68.bin"),
				"learned landmark detection model")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug2did.txt"),
				"landmark identifier to model vertex number mapping")
			("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share/model_contours.json"),
				"file with model contour indices")
			("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value("../share/sfm_3448_edge_topology.json"),
				"file with model's precomputed edge topology")
			("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
				"file with blendshapes")
			("input image,i", po::value<string>(&input_prefix)->required()->default_value("face"),
				"pre-fix name of input images")
			("extension", po::value<string>(&input_ext)->required()->default_value("bmp"),
				"path to leftside face image")
			("input-frames,n", po::value<int>(&input_frames)->required()->default_value(0),
				"path to rightside face image")
			("video",
				"Use computer camera to capture video instead of using input images")
			;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: 4dface [options]" << endl;
			cout << desc;
			return EXIT_FAILURE;
		}
		if (vm.count("video"))
			useVideo = true;
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	// Load the Morphable Model and the LandmarkMapper:
	morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(modelfile.string());
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

	// Load the landmark detector:
	rcr::detection_model rcr_model;
	try {
		rcr_model = rcr::load_detection_model(landmarkdetector.string());
	}
	catch (const cereal::Exception& e) {
		cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the face detector from OpenCV:
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(facedetector.string()))
	{
		cout << "Error loading the face detector " << facedetector << "." << endl;
		return EXIT_FAILURE;
	}

	//Load blendshapes
	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	//Load model topology
	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	//Read input image or open video capture
	cv::Mat inputImage;
	cv::VideoCapture inputVideo;
	std::vector<cv::Mat> InputImages;

	if(useVideo)
	{
		inputVideo.open(0);
		if(!inputVideo.isOpened())
		{
			std::cerr << "Couldn't open camera 0!" << endl;
			return EXIT_FAILURE;
		}
	}
	else
	{
		try
		{
			if (input_frames <= 0)
			{
				std::cerr << "Input image number must be positive integer!!" << endl;
				return EXIT_FAILURE;
			}

			for (int i = 1; i <= input_frames; i++)
			{
				char filename[256];
				sprintf(filename, ".\\images\\%s%d.%s", input_prefix, i, input_ext);
				inputImage = cv::imread(filename);

				std::string winName = "Input Image";
				cv::namedWindow(winName, 1);

				if (!inputImage.empty())
					InputImages.push_back(inputImage);
				else
					std::cerr << "No valid image input: " << filename << i << "!" << endl;


				cv::imshow(winName, InputImages[i-1]);
				cv::waitKey();
			}

			if (InputImages.size() < 1)
			{
				std::cerr << "Less than 1 input images are found, program will quit!" << endl;
				return EXIT_FAILURE;
			}
		}
		catch(std::exception ex)
		{
			cout << "Exception during reading input images: " << ex.what() << endl;
			return EXIT_FAILURE;
		}
	}



	cv::namedWindow("video", 1);
	cv::namedWindow("render", 1);

	Mat frame, unmodified_frame;

	bool have_face = false;
	rcr::LandmarkCollection<Vec2f> current_landmarks;
	Rect current_facebox;
	WeightedIsomapAveraging isomap_averaging(60.f); // merge all triangles that are facing <60?towards the camera
	PcaCoefficientMerging pca_shape_merging;

	Mat merged_isomap;

	fitting::RenderingParameters rendering_params;
	vector<float> shape_coefficients, blendshape_coefficients;
	vector<Vec2f> image_points;
	render::Mesh mesh;


	int loop = 0;
	while(loop < InputImages.size()||useVideo)
	{
		if(!useVideo)
		{
			frame = InputImages[loop].clone(); // get a new input image
			loop++;
			if (frame.empty()) { // stop if we're at the end of the video
				continue;
			}
		}
		else
		{
			inputVideo >> frame;
			if(frame.empty())
				break;
		}

		// We do a quick check if the current face's width is <= 50 pixel. If it is, we re-initialise the tracking with the face detector.
		if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
			cout << "Reinitialising because the face bounding-box width is <= 50 px" << endl;
			have_face = false;
		}

		unmodified_frame = frame.clone();

		if (!have_face) {
			// Run the face detector and obtain the initial estimate using the mean landmarks:
			vector<Rect> detected_faces;
			face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
			if (detected_faces.empty()) {
				cv::imshow("video", frame);
				cv::waitKey(30);
				continue;
			}
			cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
			// Rescale the V&J facebox to make it more like an ibug-facebox:
			// (also make sure the bounding box is square, V&J's is square)
			Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

			current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);
			rcr::draw_landmarks(frame, current_landmarks, { 0, 0, 255 }); // red, initial landmarks, {b,g,r}

			//In order to improve performance
			//when use video as input, set have_face to true to enable face tracking
			//instead of do face detection for each frame.
			if(useVideo)
				have_face = true;
		}
		else {
			// We already have a face - track and initialise using the enclosing bounding
			// box from the landmarks from the last frame:
			auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
			enclosing_bbox = make_bbox_square(enclosing_bbox);
			current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
			rcr::draw_landmarks(frame, current_landmarks, { 255, 0, 0 }); // blue, the new optimised landmarks, {b,g,r}
		}

		// Fit the 3DMM:
		std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes, rcr_to_eos_landmark_collection(current_landmarks), landmark_mapper, unmodified_frame.cols, unmodified_frame.rows, edge_topology, ibug_contour, model_contour, 3, 9, 15.0f, boost::none, shape_coefficients, blendshape_coefficients, image_points);

		// Draw the 3D pose of the face:
		draw_axes_topright(glm::eulerAngles(rendering_params.get_rotation())[0], glm::eulerAngles(rendering_params.get_rotation())[1], glm::eulerAngles(rendering_params.get_rotation())[2], frame);

		// Wireframe rendering of mesh of this frame (non-averaged):
		draw_wireframe(frame, mesh, rendering_params.get_modelview(), rendering_params.get_projection(), fitting::get_opencv_viewport(frame.cols, frame.rows));

		// Extract the texture using the fitted mesh from this frame:
		Mat affine_cam = fitting::get_3x4_affine_camera_matrix(rendering_params, frame.cols, frame.rows);
		Mat isomap = render::extract_texture(mesh, affine_cam, unmodified_frame, true, render::TextureInterpolation::NearestNeighbour, 512);

		// Merge the isomaps - add the current one to the already merged ones:
		merged_isomap = isomap_averaging.add_and_merge(isomap);
		// Same for the shape:
		shape_coefficients = pca_shape_merging.add_and_merge(shape_coefficients);
		auto merged_shape = morphable_model.get_shape_model().draw_sample(shape_coefficients) + morphablemodel::to_matrix(blendshapes) * Mat(blendshape_coefficients);
		render::Mesh merged_mesh = morphablemodel::sample_to_mesh(merged_shape, morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());

		// Render the model in a separate window using the estimated pose, shape and merged texture:
		Mat rendering;
		auto modelview_no_translation = rendering_params.get_modelview();
		modelview_no_translation[3][0] = 0;
		modelview_no_translation[3][1] = 0;
		std::tie(rendering, std::ignore) = render::render(merged_mesh, modelview_no_translation, glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, render::create_mipmapped_texture(merged_isomap), true, false, false);
		cv::imshow("render", rendering);

		cv::imshow("video", frame);
		auto key = cv::waitKey(30);
		
		if (useVideo)
		{
			if (key == 'q') break;
			if (key == 'r')
			{
				have_face = false;
				isomap_averaging = WeightedIsomapAveraging(60.f);
			}
			if (key == 's')
			{
				render::Mesh neutral_expression = morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(shape_coefficients), morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
				render::write_textured_obj(neutral_expression, ".\\models\\current_merged.obj");
				cv::imwrite(".\\models\\current_merged.isomap.png", merged_isomap);
				cout << "save model under the folder .\\models\\" << endl;
			}
		}
	}

	render::Mesh neutral_expression = morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(shape_coefficients), morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
	render::write_textured_obj(neutral_expression, ".\\models\\current_merged.obj");
	cv::imwrite(".\\models\\current_merged.isomap.png", merged_isomap);
	cout << "save model under the folder .\\models\\" << endl;

	cv::waitKey();
	return EXIT_SUCCESS;
};

/**
 * @brief Draws 3D axes onto the top-right corner of the image. The
 * axes are oriented corresponding to the given angles.
 *
 * @param[in] r_x Pitch angle, in radians.
 * @param[in] r_y Yaw angle, in radians.
 * @param[in] r_z Roll angle, in radians.
 * @param[in] image The image to draw onto.
 */
void draw_axes_topright(float r_x, float r_y, float r_z, cv::Mat image)
{
	const glm::vec3 origin(0.0f, 0.0f, 0.0f);
	const glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
	const glm::vec3 y_axis(0.0f, 1.0f, 0.0f);
	const glm::vec3 z_axis(0.0f, 0.0f, 1.0f);

	const auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
	const auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
	const auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
	const auto modelview = rot_mtx_z * rot_mtx_x * rot_mtx_y;

	const auto viewport = fitting::get_opencv_viewport(image.cols, image.rows);
	const float aspect = static_cast<float>(image.cols) / image.rows;
	const auto ortho_projection = glm::ortho(-3.0f * aspect, 3.0f * aspect, -3.0f, 3.0f);
	const auto translate_topright = glm::translate(glm::mat4(1.0f), glm::vec3(0.7f, 0.65f, 0.0f));
	const auto o_2d = glm::project(origin, modelview, translate_topright * ortho_projection, viewport);
	const auto x_2d = glm::project(x_axis, modelview, translate_topright * ortho_projection, viewport);
	const auto y_2d = glm::project(y_axis, modelview, translate_topright * ortho_projection, viewport);
	const auto z_2d = glm::project(z_axis, modelview, translate_topright * ortho_projection, viewport);
	cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ x_2d.x, x_2d.y }, { 0, 0, 255 });
	cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ y_2d.x, y_2d.y }, { 0, 255, 0 });
	cv::line(image, cv::Point2f{ o_2d.x, o_2d.y }, cv::Point2f{ z_2d.x, z_2d.y }, { 255, 0, 0 });
};
