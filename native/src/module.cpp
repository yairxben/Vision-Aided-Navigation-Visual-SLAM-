// generic_slam._native: C++ kernels for the tracking frontend.
//
// Every kernel reproduces the Python path bit for bit (same OpenCV build,
// same arithmetic order), so a run with frontend.backend = "native" writes
// exactly the artifacts of the Python backend. Notes on the arithmetic:
//  * numpy's matmul goes through OpenBLAS: matrix @ matrix rounds as
//    fma(a2, b2, fma(a1, b1, a0 * b0)); matrix @ vector (3x1) rounds as
//    fma(a2, b2, fma(a0, b0, a1 * b1)). Both are emulated below.
//  * Everything else is plain IEEE arithmetic, so the module is compiled with
//    -ffp-contract=off to stop the compiler from fusing a * b + c itself.
//  * RANSAC samples come from a copy of Python's random module state
//    (MT19937 + random.sample), handed back after each transition.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

// Raised where the Python path raises ContractError.
class ContractViolation : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

using ByteArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr double kProjectionEpsilon = 1e-10;  // frontend.stereo.PROJECTION_EPSILON

// ---------------------------------------------------------------- utilities
cv::Mat view_bytes(const ByteArray& array, const char* label) {
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(label) + " must be a 2-D uint8 array");
    }
    return cv::Mat(static_cast<int>(array.shape(0)), static_cast<int>(array.shape(1)), CV_8UC1,
                   const_cast<std::uint8_t*>(array.data()));
}

cv::Mat copy_doubles(const DoubleArray& array, int rows, int cols, const char* label) {
    if (array.size() != static_cast<py::ssize_t>(rows) * cols) {
        throw std::invalid_argument(std::string(label) + " has the wrong size");
    }
    cv::Mat value(rows, cols, CV_64F);
    std::memcpy(value.ptr<double>(), array.data(), sizeof(double) * rows * cols);
    return value;
}

using Mat3 = std::array<double, 9>;  // row-major
using Vec3 = std::array<double, 3>;

Mat3 to_mat3(const cv::Mat& m) {
    Mat3 out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[3 * i + j] = m.at<double>(i, j);
    return out;
}

Vec3 to_vec3(const cv::Mat& m) {
    return {m.at<double>(0), m.at<double>(1), m.at<double>(2)};
}

// numpy matmul (OpenBLAS gemm) of a 3x3 row with a 3-vector.
inline double gemm_dot(const double* row, double b0, double b1, double b2) {
    return std::fma(row[2], b2, std::fma(row[1], b1, row[0] * b0));
}

// numpy matmul (OpenBLAS gemv) of a 3x3 row with a 3x1 column.
inline double gemv_dot(const double* row, double b0, double b1, double b2) {
    return std::fma(row[2], b2, std::fma(row[0], b0, row[1] * b1));
}

// compose_right_camera: (F @ R, F @ t + f).
void compose_right(const Mat3& fixed_r, const Vec3& fixed_t, const Mat3& r, const Vec3& t,
                   Mat3& out_r, Vec3& out_t) {
    for (int i = 0; i < 3; ++i) {
        const double* row = &fixed_r[3 * i];
        for (int j = 0; j < 3; ++j) out_r[3 * i + j] = gemm_dot(row, r[j], r[3 + j], r[6 + j]);
        out_t[i] = gemv_dot(row, t[0], t[1], t[2]) + fixed_t[i];
    }
}

template <typename T>
py::array_t<T> vector_array(const std::vector<T>& values) {
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) std::memcpy(out.mutable_data(), values.data(), sizeof(T) * values.size());
    return out;
}

// ------------------------------------------------------- Python's random
// CPython's _random.Random (MT19937) plus random.sample / _randbelow (3.8).
class PythonRandom {
   public:
    explicit PythonRandom(const std::vector<std::uint64_t>& state) {
        if (state.size() != kN + 1) throw std::invalid_argument("random state must have 625 values");
        for (int i = 0; i < kN; ++i) mt_[i] = static_cast<std::uint32_t>(state[i]);
        index_ = static_cast<int>(state[kN]);
        if (index_ < 0 || index_ > kN) throw std::invalid_argument("random state index is invalid");
    }

    std::vector<std::uint64_t> state() const {
        std::vector<std::uint64_t> out(mt_.begin(), mt_.end());
        out.push_back(static_cast<std::uint64_t>(index_));
        return out;
    }

    // random.sample(range(n), k): the chosen positions, in draw order.
    std::vector<std::size_t> sample(std::size_t n, std::size_t k) {
        if (k > n) throw ContractViolation("Sample larger than population or is negative");
        std::vector<std::size_t> result(k);
        std::size_t setsize = 21;
        if (k > 5) {
            setsize += static_cast<std::size_t>(
                std::pow(4.0, std::ceil(std::log(static_cast<double>(k * 3)) / std::log(4.0))));
        }
        if (n <= setsize) {
            std::vector<std::size_t> pool(n);
            for (std::size_t i = 0; i < n; ++i) pool[i] = i;
            for (std::size_t i = 0; i < k; ++i) {
                const std::size_t j = randbelow(n - i);
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            std::unordered_set<std::size_t> selected;
            for (std::size_t i = 0; i < k; ++i) {
                std::size_t j = randbelow(n);
                while (selected.count(j)) j = randbelow(n);
                selected.insert(j);
                result[i] = j;
            }
        }
        return result;
    }

   private:
    static constexpr int kN = 624;
    static constexpr int kM = 397;

    std::uint32_t genrand_uint32() {
        static const std::uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        std::uint32_t y;
        if (index_ >= kN) {
            int kk;
            for (kk = 0; kk < kN - kM; ++kk) {
                y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + kM] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < kN - 1; ++kk) {
                y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + (kM - kN)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (mt_[kN - 1] & 0x80000000U) | (mt_[0] & 0x7fffffffU);
            mt_[kN - 1] = mt_[kM - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index_ = 0;
        }
        y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    std::size_t randbelow(std::size_t n) {
        int bits = 0;
        for (std::size_t v = n; v; v >>= 1) ++bits;  // n.bit_length()
        if (bits > 32) throw ContractViolation("population too large for the native sampler");
        std::size_t r = genrand_uint32() >> (32 - bits);
        while (r >= n) r = genrand_uint32() >> (32 - bits);
        return r;
    }

    std::array<std::uint32_t, kN> mt_{};
    int index_ = kN;
};

// ------------------------------------------------------------ stereo frame
struct StereoFrame {
    std::vector<cv::KeyPoint> left;
    std::vector<cv::KeyPoint> right;
    cv::Mat left_descriptors;
    cv::Mat right_descriptors;
    std::vector<cv::DMatch> inliers;  // ascending queryIdx, one per left feature
    std::size_t raw_match_count = 0;
};

cv::Ptr<cv::AKAZE> make_akaze(double threshold) {
    // cv2.AKAZE_create(threshold=...) with every other argument at its default.
    return cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, static_cast<float>(threshold), 4, 4,
                             cv::KAZE::DIFF_PM_G2);
}

std::shared_ptr<StereoFrame> extract_stereo(const ByteArray& left_image, const ByteArray& right_image,
                                            double threshold, double maximum_vertical_deviation_px,
                                            bool require_positive_disparity) {
    const cv::Mat left_view = view_bytes(left_image, "left stereo image");
    const cv::Mat right_view = view_bytes(right_image, "right stereo image");
    auto frame = std::make_shared<StereoFrame>();
    py::gil_scoped_release release;
    // The right image is detected on a second thread, as in the Python frontend.
    auto right = std::async(std::launch::async, [&] {
        make_akaze(threshold)->detectAndCompute(right_view, cv::noArray(), frame->right,
                                                frame->right_descriptors);
    });
    make_akaze(threshold)->detectAndCompute(left_view, cv::noArray(), frame->left,
                                            frame->left_descriptors);
    right.get();
    if (frame->left_descriptors.empty()) {
        throw ContractViolation("feature detector produced no descriptors for left stereo image");
    }
    if (frame->right_descriptors.empty()) {
        throw ContractViolation("feature detector produced no descriptors for right stereo image");
    }
    std::vector<cv::DMatch> matches;
    cv::BFMatcher(cv::NORM_HAMMING, false).match(frame->left_descriptors, frame->right_descriptors,
                                                 matches);
    frame->raw_match_count = matches.size();
    for (const cv::DMatch& match : matches) {
        const cv::Point2f& l = frame->left[static_cast<std::size_t>(match.queryIdx)].pt;
        const cv::Point2f& r = frame->right[static_cast<std::size_t>(match.trainIdx)].pt;
        const bool rectified = std::fabs(static_cast<double>(l.y) - static_cast<double>(r.y)) <=
                               maximum_vertical_deviation_px;
        const bool positive = static_cast<double>(l.x) - static_cast<double>(r.x) > 0.0;
        if (rectified && (positive || !require_positive_disparity)) frame->inliers.push_back(match);
    }
    std::stable_sort(frame->inliers.begin(), frame->inliers.end(),
                     [](const cv::DMatch& a, const cv::DMatch& b) { return a.queryIdx < b.queryIdx; });
    return frame;
}

// PreparedStereoFrame.links: u_left, u_right, mean of the two rows.
py::array_t<double> stereo_links(const StereoFrame& frame) {
    const auto count = static_cast<py::ssize_t>(frame.inliers.size());
    py::array_t<double> out({count, static_cast<py::ssize_t>(3)});
    auto rows = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < count; ++i) {
        const cv::DMatch& m = frame.inliers[static_cast<std::size_t>(i)];
        const cv::Point2f& l = frame.left[static_cast<std::size_t>(m.queryIdx)].pt;
        const cv::Point2f& r = frame.right[static_cast<std::size_t>(m.trainIdx)].pt;
        rows(i, 0) = l.x;
        rows(i, 1) = r.x;
        rows(i, 2) = (static_cast<double>(l.y) + static_cast<double>(r.y)) / 2;
    }
    return out;
}

py::array_t<std::int64_t> stereo_rows(const StereoFrame& frame) {
    std::vector<std::int64_t> rows;
    rows.reserve(frame.inliers.size());
    for (const cv::DMatch& m : frame.inliers) rows.push_back(m.queryIdx);
    return vector_array(rows);
}

// --------------------------------------------------------------- transition
struct Correspondence {
    int previous_stereo;
    int current_stereo;
    int temporal;
};

struct CameraModel {
    std::array<Mat3, 4> r;
    std::array<Vec3, 4> t;
};

struct RansacSettings {
    double success_probability;
    long long initial_remaining;
    std::size_t sample_size;
    double maximum_distance_px;
    std::optional<long long> maximum_attempts;
};

class Transition {
   public:
    Transition(const StereoFrame& previous, const StereoFrame& current, const cv::Mat& intrinsic,
               const std::array<Mat3, 2>& previous_r, const std::array<Vec3, 2>& previous_t,
               const Mat3& fixed_r, const Vec3& fixed_t)
        : previous_(previous),
          current_(current),
          intrinsic_(intrinsic),
          k_(to_mat3(intrinsic)),
          previous_r_(previous_r),
          previous_t_(previous_t),
          fixed_r_(fixed_r),
          fixed_t_(fixed_t) {}

    // triangulate_stereo_matches on the previous frame's stereo inliers.
    void triangulate(const cv::Mat& projection_left, const cv::Mat& projection_right) {
        const int count = static_cast<int>(previous_.inliers.size());
        cloud_.assign(static_cast<std::size_t>(count), Vec3{});
        if (!count) return;
        cv::Mat left(2, count, CV_64F), right(2, count, CV_64F);
        for (int i = 0; i < count; ++i) {
            const cv::DMatch& m = previous_.inliers[static_cast<std::size_t>(i)];
            const cv::Point2f& l = previous_.left[static_cast<std::size_t>(m.queryIdx)].pt;
            const cv::Point2f& r = previous_.right[static_cast<std::size_t>(m.trainIdx)].pt;
            left.at<double>(0, i) = l.x;
            left.at<double>(1, i) = l.y;
            right.at<double>(0, i) = r.x;
            right.at<double>(1, i) = r.y;
        }
        cv::Mat homogeneous;
        cv::triangulatePoints(projection_left, projection_right, left, right, homogeneous);
        if (homogeneous.type() != CV_64F) throw ContractViolation("triangulation output is not float64");
        for (int i = 0; i < count; ++i) {
            const double w = homogeneous.at<double>(3, i) + kProjectionEpsilon;
            cloud_[static_cast<std::size_t>(i)] = {homogeneous.at<double>(0, i) / w,
                                                   homogeneous.at<double>(1, i) / w,
                                                   homogeneous.at<double>(2, i) / w};
        }
    }

    // FeatureFrontend.reciprocal_matches_on_rows on both frames' stereo rows.
    void match_stereo_rows() {
        std::vector<int> back_rows, front_rows;
        for (const cv::DMatch& m : previous_.inliers) back_rows.push_back(m.queryIdx);
        for (const cv::DMatch& m : current_.inliers) front_rows.push_back(m.queryIdx);
        if (back_rows.empty() || front_rows.empty()) return;
        const cv::Mat back = select_rows(previous_.left_descriptors, back_rows);
        const cv::Mat front = select_rows(current_.left_descriptors, front_rows);
        std::vector<cv::DMatch> forward, reverse;
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        matcher.match(back, front, forward);
        matcher.match(front, back, reverse);
        std::vector<int> reverse_map(static_cast<std::size_t>(front.rows), -1);
        for (const cv::DMatch& m : reverse) reverse_map[static_cast<std::size_t>(m.queryIdx)] = m.trainIdx;
        std::vector<cv::DMatch> kept;
        for (const cv::DMatch& m : forward) {
            if (reverse_map[static_cast<std::size_t>(m.trainIdx)] == m.queryIdx) kept.push_back(m);
        }
        std::stable_sort(kept.begin(), kept.end(),
                         [](const cv::DMatch& a, const cv::DMatch& b) { return a.queryIdx < b.queryIdx; });
        for (const cv::DMatch& m : kept) {
            reciprocal_query_.push_back(back_rows[static_cast<std::size_t>(m.queryIdx)]);
            reciprocal_train_.push_back(front_rows[static_cast<std::size_t>(m.trainIdx)]);
            reciprocal_distance_.push_back(m.distance);
        }
    }

    // build_four_view_consensus.
    void build_consensus() {
        std::vector<int> previous_position(previous_.left.size(), -1);
        std::vector<int> current_position(current_.left.size(), -1);
        for (std::size_t i = 0; i < previous_.inliers.size(); ++i)
            previous_position[static_cast<std::size_t>(previous_.inliers[i].queryIdx)] = static_cast<int>(i);
        for (std::size_t i = 0; i < current_.inliers.size(); ++i)
            current_position[static_cast<std::size_t>(current_.inliers[i].queryIdx)] = static_cast<int>(i);
        for (std::size_t i = 0; i < reciprocal_query_.size(); ++i) {
            const int p = previous_position[static_cast<std::size_t>(reciprocal_query_[i])];
            const int c = current_position[static_cast<std::size_t>(reciprocal_train_[i])];
            if (p >= 0 && c >= 0) consensus_.push_back({p, c, static_cast<int>(i)});
        }
        const std::size_t n = consensus_.size();
        points_.resize(n);
        observed_.assign(n, {});
        for (std::size_t i = 0; i < n; ++i) {
            const cv::DMatch& p = previous_.inliers[static_cast<std::size_t>(consensus_[i].previous_stereo)];
            const cv::DMatch& c = current_.inliers[static_cast<std::size_t>(consensus_[i].current_stereo)];
            points_[i] = cloud_[static_cast<std::size_t>(consensus_[i].previous_stereo)];
            const cv::Point2f pts[4] = {previous_.left[static_cast<std::size_t>(p.queryIdx)].pt,
                                        previous_.right[static_cast<std::size_t>(p.trainIdx)].pt,
                                        current_.left[static_cast<std::size_t>(c.queryIdx)].pt,
                                        current_.right[static_cast<std::size_t>(c.trainIdx)].pt};
            for (int camera = 0; camera < 4; ++camera) {
                observed_[i][2 * camera] = pts[camera].x;
                observed_[i][2 * camera + 1] = pts[camera].y;
            }
        }
    }

    // run_visual_odometry_ransac (adaptive search, then strict refinement).
    void run_ransac(const RansacSettings& settings, PythonRandom& random, const py::function& required) {
        const std::size_t n = consensus_.size();
        if (n < settings.sample_size) throw ContractViolation("insufficient four-view consensus for VO RANSAC");
        long long remaining = settings.initial_remaining;
        std::optional<CameraModel> best_model;
        const auto capped = [&] {
            return settings.maximum_attempts && attempts_ >= *settings.maximum_attempts;
        };
        while (remaining > 0) {
            if (capped()) {
                if (!best_model) throw ContractViolation("VO RANSAC exhausted its experimental attempt cap");
                break;
            }
            std::vector<std::size_t> sample = random.sample(n, settings.sample_size);
            CameraModel model;
            while (true) {
                ++attempts_;
                bool success = false;
                try {
                    success = build_model(sample, model);
                } catch (const cv::Exception&) {
                    success = false;
                }
                if (success) break;
                if (capped()) {
                    if (!best_model) throw ContractViolation("VO RANSAC exhausted its experimental attempt cap");
                    model_ = *best_model;  // the Python path returns here without refinement
                    return;
                }
                sample = random.sample(n, settings.sample_size);
            }
            std::vector<std::int64_t> supporters = supporter_indices(model, settings.maximum_distance_px);
            if (supporters.size() > best_.size()) {
                best_model = model;
                best_ = std::move(supporters);
                const double outlier = 1.0 - static_cast<double>(best_.size()) / static_cast<double>(n);
                if (outlier <= 0.0) {
                    remaining = 1;
                } else {
                    py::gil_scoped_acquire acquire;
                    remaining = required(settings.success_probability, outlier, settings.sample_size)
                                    .cast<long long>();
                }
            } else {
                --remaining;
            }
        }
        if (!best_model || best_.size() < settings.sample_size) {
            throw ContractViolation("VO RANSAC produced no refinable model");
        }
        while (true) {
            std::vector<std::size_t> refinement(best_.begin(), best_.end());
            ++attempts_;
            CameraModel model;
            if (!build_model(refinement, model)) {
                // The Python path would retry the identical call forever.
                throw ContractViolation("VO RANSAC refinement PnP failed");
            }
            std::vector<std::int64_t> supporters = supporter_indices(model, settings.maximum_distance_px);
            if (supporters.size() > best_.size()) {
                best_ = std::move(supporters);
                best_model = model;
            } else {
                break;
            }
        }
        model_ = *best_model;
    }

    py::dict result(const PythonRandom& random) const {
        py::dict out;
        out["reciprocal_query"] = vector_array(reciprocal_query_);
        out["reciprocal_train"] = vector_array(reciprocal_train_);
        std::vector<double> distance(reciprocal_distance_.begin(), reciprocal_distance_.end());
        out["reciprocal_distance"] = vector_array(distance);
        const auto n = static_cast<py::ssize_t>(consensus_.size());
        py::array_t<std::int64_t> consensus({n, static_cast<py::ssize_t>(3)});
        py::array_t<double> points({n, static_cast<py::ssize_t>(3)});
        py::array_t<double> observed({static_cast<py::ssize_t>(4), static_cast<py::ssize_t>(2), n});
        auto c = consensus.mutable_unchecked<2>();
        auto p = points.mutable_unchecked<2>();
        auto o = observed.mutable_unchecked<3>();
        for (py::ssize_t i = 0; i < n; ++i) {
            const Correspondence& item = consensus_[static_cast<std::size_t>(i)];
            c(i, 0) = item.previous_stereo;
            c(i, 1) = item.current_stereo;
            c(i, 2) = item.temporal;
            for (int k = 0; k < 3; ++k) p(i, k) = points_[static_cast<std::size_t>(i)][k];
            for (int camera = 0; camera < 4; ++camera)
                for (int k = 0; k < 2; ++k) o(camera, k, i) = observed_[static_cast<std::size_t>(i)][2 * camera + k];
        }
        out["consensus"] = consensus;
        out["consensus_points"] = points;
        out["observed_pixels"] = observed;
        out["supporter_indices"] = vector_array(best_);
        py::array_t<double> rotation({3, 3}), translation({3, 1});
        std::memcpy(rotation.mutable_data(), model_.r[2].data(), sizeof(double) * 9);
        std::memcpy(translation.mutable_data(), model_.t[2].data(), sizeof(double) * 3);
        out["rotation"] = rotation;
        out["translation"] = translation;
        out["attempts"] = attempts_;
        out["random_state"] = random.state();
        return out;
    }

   private:
    static cv::Mat select_rows(const cv::Mat& descriptors, const std::vector<int>& rows) {
        cv::Mat out(static_cast<int>(rows.size()), descriptors.cols, descriptors.type());
        for (std::size_t i = 0; i < rows.size(); ++i) descriptors.row(rows[i]).copyTo(out.row(static_cast<int>(i)));
        return out;
    }

    // build_four_camera_model: EPnP on the sample, then compose the four cameras.
    bool build_model(const std::vector<std::size_t>& sample, CameraModel& model) const {
        if (sample.size() < 4) return false;  // solve_front_camera's ContractError
        const int count = static_cast<int>(sample.size());
        cv::Mat object_points(count, 3, CV_64F), image_points(count, 2, CV_64F);
        for (int i = 0; i < count; ++i) {
            const Correspondence& item = consensus_[sample[static_cast<std::size_t>(i)]];
            const Vec3& point = cloud_[static_cast<std::size_t>(item.previous_stereo)];
            for (int k = 0; k < 3; ++k) object_points.at<double>(i, k) = point[k];
            const cv::DMatch& m = current_.inliers[static_cast<std::size_t>(item.current_stereo)];
            const cv::Point2f& pixel = current_.left[static_cast<std::size_t>(m.queryIdx)].pt;
            image_points.at<double>(i, 0) = pixel.x;
            image_points.at<double>(i, 1) = pixel.y;
        }
        cv::Mat rotation_vector, translation, rotation;
        const bool success = cv::solvePnP(object_points, image_points, intrinsic_, cv::noArray(),
                                          rotation_vector, translation, false, cv::SOLVEPNP_EPNP);
        cv::Rodrigues(rotation_vector, rotation);
        if (!success) return false;
        model.r[0] = previous_r_[0];
        model.t[0] = previous_t_[0];
        model.r[1] = previous_r_[1];
        model.t[1] = previous_t_[1];
        model.r[2] = to_mat3(rotation);
        model.t[2] = to_vec3(translation);
        compose_right(fixed_r_, fixed_t_, model.r[2], model.t[2], model.r[3], model.t[3]);
        return true;
    }

    // supporter_indices: consensus rows within the pixel bound in all four cameras.
    std::vector<std::int64_t> supporter_indices(const CameraModel& model, double maximum_distance_px) const {
        std::vector<std::int64_t> out;
        const double* k = k_.data();
        for (std::size_t i = 0; i < points_.size(); ++i) {
            const Vec3& p = points_[i];
            bool inside = true;
            for (int camera = 0; camera < 4 && inside; ++camera) {
                const double* r = model.r[camera].data();
                const Vec3& t = model.t[camera];
                const double y0 = gemm_dot(r, p[0], p[1], p[2]) + t[0];
                const double y1 = gemm_dot(r + 3, p[0], p[1], p[2]) + t[1];
                const double y2 = gemm_dot(r + 6, p[0], p[1], p[2]) + t[2];
                const double z0 = gemm_dot(k, y0, y1, y2);
                const double z1 = gemm_dot(k + 3, y0, y1, y2);
                const double z2 = gemm_dot(k + 6, y0, y1, y2);
                const double w = z2 + kProjectionEpsilon;
                const double du = observed_[i][2 * camera] - z0 / w;
                const double dv = observed_[i][2 * camera + 1] - z1 / w;
                inside = std::sqrt(du * du + dv * dv) <= maximum_distance_px;
            }
            if (inside) out.push_back(static_cast<std::int64_t>(i));
        }
        return out;
    }

    const StereoFrame& previous_;
    const StereoFrame& current_;
    cv::Mat intrinsic_;
    Mat3 k_;
    std::array<Mat3, 2> previous_r_;
    std::array<Vec3, 2> previous_t_;
    Mat3 fixed_r_;
    Vec3 fixed_t_;
    std::vector<Vec3> cloud_;
    std::vector<std::int64_t> reciprocal_query_, reciprocal_train_;
    std::vector<float> reciprocal_distance_;
    std::vector<Correspondence> consensus_;
    std::vector<Vec3> points_;
    std::vector<std::array<double, 8>> observed_;  // (u, v) in each of the four cameras
    std::vector<std::int64_t> best_;
    CameraModel model_{};
    long long attempts_ = 0;
};

py::dict track_transition(const StereoFrame& previous, const StereoFrame& current,
                          const DoubleArray& projection_left, const DoubleArray& projection_right,
                          const DoubleArray& intrinsic, const DoubleArray& previous_left_rotation,
                          const DoubleArray& previous_left_translation,
                          const DoubleArray& previous_right_rotation,
                          const DoubleArray& previous_right_translation,
                          const DoubleArray& fixed_right_rotation,
                          const DoubleArray& fixed_right_translation, double success_probability,
                          long long initial_remaining, std::size_t sample_size,
                          double maximum_distance_px, std::optional<long long> maximum_attempts,
                          const py::function& required_hypothesis_count,
                          const std::vector<std::uint64_t>& random_state) {
    const cv::Mat k = copy_doubles(intrinsic, 3, 3, "intrinsic");
    const cv::Mat p_left = copy_doubles(projection_left, 3, 4, "left projection");
    const cv::Mat p_right = copy_doubles(projection_right, 3, 4, "right projection");
    const std::array<Mat3, 2> previous_r = {
        to_mat3(copy_doubles(previous_left_rotation, 3, 3, "previous left rotation")),
        to_mat3(copy_doubles(previous_right_rotation, 3, 3, "previous right rotation"))};
    const std::array<Vec3, 2> previous_t = {
        to_vec3(copy_doubles(previous_left_translation, 3, 1, "previous left translation")),
        to_vec3(copy_doubles(previous_right_translation, 3, 1, "previous right translation"))};
    const Mat3 fixed_r = to_mat3(copy_doubles(fixed_right_rotation, 3, 3, "fixed right rotation"));
    const Vec3 fixed_t = to_vec3(copy_doubles(fixed_right_translation, 3, 1, "fixed right translation"));
    PythonRandom random(random_state);
    Transition transition(previous, current, k, previous_r, previous_t, fixed_r, fixed_t);
    {
        py::gil_scoped_release release;
        transition.triangulate(p_left, p_right);
        transition.match_stereo_rows();
        transition.build_consensus();
        transition.run_ransac({success_probability, initial_remaining, sample_size,
                               maximum_distance_px, maximum_attempts},
                              random, required_hypothesis_count);
    }
    return transition.result(random);
}

// ------------------------------------------------------ Phase 1 primitives
py::tuple akaze_detect_and_compute(const ByteArray& image, double threshold) {
    const cv::Mat view = view_bytes(image, "image");
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    {
        py::gil_scoped_release release;
        make_akaze(threshold)->detectAndCompute(view, cv::noArray(), keypoints, descriptors);
    }
    const auto count = static_cast<py::ssize_t>(keypoints.size());
    py::array_t<double> points({count, static_cast<py::ssize_t>(7)});
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
    if (descriptors.total()) std::memcpy(bytes.mutable_data(), descriptors.ptr(), descriptors.total());
    return py::make_tuple(points, bytes);
}

py::tuple hamming_match(const ByteArray& query, const ByteArray& train) {
    const cv::Mat query_view = view_bytes(query, "query descriptors");
    const cv::Mat train_view = view_bytes(train, "train descriptors");
    std::vector<cv::DMatch> matches;
    {
        py::gil_scoped_release release;
        cv::BFMatcher(cv::NORM_HAMMING, false).match(query_view, train_view, matches);
    }
    std::vector<std::int64_t> q, t;
    std::vector<float> d;
    for (const cv::DMatch& m : matches) {
        q.push_back(m.queryIdx);
        t.push_back(m.trainIdx);
        d.push_back(m.distance);
    }
    return py::make_tuple(vector_array(q), vector_array(t), vector_array(d));
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "C++ kernels for the generic_slam tracking frontend";
    py::register_exception<ContractViolation>(module, "ContractViolation");
    module.def("opencv_version", [] { return std::string(CV_VERSION); });
    module.def("opencv_build_information", [] { return cv::getBuildInformation(); });

    py::class_<StereoFrame, std::shared_ptr<StereoFrame>>(module, "StereoFrame")
        .def_property_readonly("left_feature_count", [](const StereoFrame& f) { return f.left.size(); })
        .def_property_readonly("right_feature_count", [](const StereoFrame& f) { return f.right.size(); })
        .def_property_readonly("stereo_row_count", [](const StereoFrame& f) { return f.inliers.size(); })
        .def_property_readonly("raw_match_count", [](const StereoFrame& f) { return f.raw_match_count; })
        .def("links", &stereo_links, "(rows, 3): u_left, u_right, mean row, one row per stereo inlier")
        .def("stereo_rows", &stereo_rows, "Ascending left-feature index of each stereo inlier");

    module.def("extract_stereo", &extract_stereo, py::arg("left_image"), py::arg("right_image"),
               py::arg("threshold"), py::arg("maximum_vertical_deviation_px"),
               py::arg("require_positive_disparity"),
               "AKAZE on both images and the global-nearest stereo matches that pass the geometry.");
    module.def("track_transition", &track_transition, py::arg("previous"), py::arg("current"),
               py::arg("projection_left"), py::arg("projection_right"), py::arg("intrinsic"),
               py::arg("previous_left_rotation"), py::arg("previous_left_translation"),
               py::arg("previous_right_rotation"), py::arg("previous_right_translation"),
               py::arg("fixed_right_rotation"), py::arg("fixed_right_translation"),
               py::arg("success_probability"), py::arg("initial_remaining"), py::arg("sample_size"),
               py::arg("maximum_distance_px"), py::arg("maximum_attempts"),
               py::arg("required_hypothesis_count"), py::arg("random_state"),
               "Triangulation, reciprocal matching on stereo rows, four-view consensus and VO RANSAC.");
    module.def("akaze_detect_and_compute", &akaze_detect_and_compute, py::arg("image"),
               py::arg("threshold"),
               "AKAZE keypoints (N, 7: x, y, size, angle, response, octave, class_id) "
               "and MLDB descriptors (N, 61), as cv2.AKAZE_create(threshold=...).");
    module.def("hamming_match", &hamming_match, py::arg("query"), py::arg("train"),
               "cv2.BFMatcher(NORM_HAMMING).match: (query_index, train_index, distance).");
}
