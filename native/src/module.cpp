// generic_slam._native: C++ kernels for the tracking frontend.
//
// Phase 1 exposes the OpenCV primitives the Python frontend uses, so their
// outputs can be proven identical to pip's cv2 before any logic moves here.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using ImageArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

// Keypoint columns: x, y, size, angle, response, octave, class_id.
constexpr int kKeypointColumns = 7;

cv::Mat view_image(const ImageArray& image) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("image must be a 2-D uint8 array");
    }
    return cv::Mat(static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)), CV_8UC1,
                   const_cast<std::uint8_t*>(image.data()));
}

cv::Mat view_descriptors(const ImageArray& descriptors) {
    if (descriptors.ndim() != 2) {
        throw std::invalid_argument("descriptors must be a 2-D uint8 array");
    }
    return cv::Mat(static_cast<int>(descriptors.shape(0)), static_cast<int>(descriptors.shape(1)),
                   CV_8UC1, const_cast<std::uint8_t*>(descriptors.data()));
}

py::tuple akaze_detect_and_compute(const ImageArray& image, double threshold) {
    const cv::Mat view = view_image(image);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    {
        py::gil_scoped_release release;
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create(
            cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, static_cast<float>(threshold), 4, 4,
            cv::KAZE::DIFF_PM_G2);
        detector->detectAndCompute(view, cv::noArray(), keypoints, descriptors);
    }
    const auto count = static_cast<py::ssize_t>(keypoints.size());
    py::array_t<double> points({count, static_cast<py::ssize_t>(kKeypointColumns)});
    auto rows = points.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < count; ++i) {
        const cv::KeyPoint& k = keypoints[static_cast<std::size_t>(i)];
        rows(i, 0) = k.pt.x;
        rows(i, 1) = k.pt.y;
        rows(i, 2) = k.size;
        rows(i, 3) = k.angle;
        rows(i, 4) = k.response;
        rows(i, 5) = k.octave;
        rows(i, 6) = k.class_id;
    }
    py::array_t<std::uint8_t> bytes({static_cast<py::ssize_t>(descriptors.rows),
                                     static_cast<py::ssize_t>(descriptors.cols)});
    if (descriptors.total()) {
        std::memcpy(bytes.mutable_data(), descriptors.ptr(), descriptors.total());
    }
    return py::make_tuple(points, bytes);
}

// cv::BFMatcher(NORM_HAMMING).match: best train row per query row.
py::tuple hamming_match(const ImageArray& query, const ImageArray& train) {
    const cv::Mat query_view = view_descriptors(query);
    const cv::Mat train_view = view_descriptors(train);
    std::vector<cv::DMatch> matches;
    {
        py::gil_scoped_release release;
        cv::BFMatcher(cv::NORM_HAMMING, false).match(query_view, train_view, matches);
    }
    const auto count = static_cast<py::ssize_t>(matches.size());
    py::array_t<std::int64_t> query_index(count), train_index(count);
    py::array_t<float> distance(count);
    auto q = query_index.mutable_unchecked<1>();
    auto t = train_index.mutable_unchecked<1>();
    auto d = distance.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < count; ++i) {
        const cv::DMatch& m = matches[static_cast<std::size_t>(i)];
        q(i) = m.queryIdx;
        t(i) = m.trainIdx;
        d(i) = m.distance;
    }
    return py::make_tuple(query_index, train_index, distance);
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "C++ kernels for the generic_slam tracking frontend";
    module.def("opencv_version", [] { return std::string(CV_VERSION); });
    module.def("opencv_build_information", [] { return cv::getBuildInformation(); });
    module.def("akaze_detect_and_compute", &akaze_detect_and_compute, py::arg("image"),
               py::arg("threshold"),
               "AKAZE keypoints (N, 7: x, y, size, angle, response, octave, class_id) "
               "and MLDB descriptors (N, 61), as cv2.AKAZE_create(threshold=...).");
    module.def("hamming_match", &hamming_match, py::arg("query"), py::arg("train"),
               "cv2.BFMatcher(NORM_HAMMING).match: (query_index, train_index, distance).");
}
