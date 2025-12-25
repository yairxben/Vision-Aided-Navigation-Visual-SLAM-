import pickle

import gtsam
import numpy as np
from gtsam import symbol
from tracking_database import TrackingDB
from LinkLoop import LinkLoop

from algorithms_library import create_calib_mat_gtsam, create_ext_matrix_gtsam, triangulate_gtsam

EPSILON = 1.5

LAND_MARK_SYM = "q"
CAMERA_SYM = "c"
P3D_MAX_DIST = 80


class BundleWindow:

    def __init__(self, db, first_key_frame_id=None, last_key_frame_id=None, all_frames_between=True,
                 little_bundle_track=None):

        self.db = db
        self.__first_key_frame_id = first_key_frame_id
        self.__last_key_frame_id = last_key_frame_id
        self.__all_frames_between = all_frames_between
        if first_key_frame_id is not None and last_key_frame_id:
            if all_frames_between:
                self.__bundle_frames = range(first_key_frame_id, last_key_frame_id + 1)
                self.__bundle_len = 1 + last_key_frame_id - first_key_frame_id
            else:
                self.__bundle_frames = [first_key_frame_id, last_key_frame_id]
                self.__bundle_len = 2
                self.__little_bundle_track = little_bundle_track

        self.optimizer = None
        self.__initial_estimate = gtsam.Values()
        self.__optimized_values = None
        self.__camera_sym = set()
        self.__landmark_sym = set()
        self.__tracks = set()
        self.__transformations = []
        self.db = db
        self.graph = gtsam.NonlinearFactorGraph()
        # self._graph_data = []
        # self._initial_estimate_data = {}
        # self._optimized_values_data = {}
    def graph(self):
        return self.graph

    def get_transformations(self):
        return self.__transformations

    def add_trans(self, T):
        self.__transformations.append(T)


    def add_cam_factor(self, numSymbol=0, current_Rt_gtsam=gtsam.Pose3(), pose_uncertainty=gtsam.Pose3()):
        # Initial pose
        camera_symbol_gtsam = symbol(CAMERA_SYM, numSymbol)
        factor = gtsam.PriorFactorPose3(camera_symbol_gtsam, current_Rt_gtsam, pose_uncertainty)
        self.__camera_sym.add(camera_symbol_gtsam)
        self.graph.add(factor)

    def get_last_frame_symbol(self, loop=False):
        if loop:
            return symbol(CAMERA_SYM, 1)
        return symbol(CAMERA_SYM, self.__last_key_frame_id - self.__first_key_frame_id)
        # return self.__camera_sym[-1]
    def cam_insert(self, numSymbol=0, current_Rt_gtsam=gtsam.Pose3()):
        camera_symbol_gtsam = symbol(CAMERA_SYM, numSymbol)
        self.__initial_estimate.insert(camera_symbol_gtsam, current_Rt_gtsam)
        self.__camera_sym.add(camera_symbol_gtsam)
    def save(self, base_filename):
        """Save the window to a file."""
        # with open(base_filename, 'wb') as fileHandler:
        #     pickle.dump(self, fileHandler)
        graph_str = self.graph.serialize()
        #dict all the fields to searialize to pickle

        data = {
            "first_key_frame_id": self.__first_key_frame_id,
            "last_key_frame_id": self.__last_key_frame_id,
            "all_frames_between": self.__all_frames_between,
            "bundle_len": self.__bundle_len,
            "camera_sym": self.__camera_sym,
            "landmark_sym": self.__landmark_sym,
            "tracks": self.__tracks,
            "transformations": self.__transformations,
            # "db": self.db,
            "graph": graph_str,
            "initial_estimate": self.__initial_estimate,
            "optimized_values": self.__optimized_values
        }
        filename = base_filename + '.pkl'
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print('Bundle Window saved to ', filename)

    def load(self, base_filename):
        filename = base_filename + '.pkl'
        with open(filename, "rb") as file:
            data = pickle.load(file)
            self.__first_key_frame_id = data["first_key_frame_id"]
            self.__last_key_frame_id = data["last_key_frame_id"]
            self.__all_frames_between = data["all_frames_between"]
            self.__bundle_len = data["bundle_len"]
            self.__camera_sym = data["camera_sym"]
            self.__landmark_sym = data["landmark_sym"]
            self.__tracks = data["tracks"]
            self.__transformations = data["transformations"]
            graph_data = data["graph"]
            # self.db = data["db"]
            self.graph.deserialize(graph_data)
            self.__initial_estimate = data["initial_estimate"]
            self.__optimized_values = data["optimized_values"]
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.__initial_estimate)

        # print('Bundle Window loaded from ', filename)

    def marginals(self):
        #print the values i \

        return gtsam.Marginals(self.graph, self.get_optimized_values())
    def get_optimized_values(self):
        return self.__optimized_values
    def create_factor_graph(self):

        self.compose_transformations()

        calib_mat_gtsam = create_calib_mat_gtsam()

        if self.__all_frames_between:
            print(f"Creating factor graph for frames from {self.__first_key_frame_id} to {self.__last_key_frame_id}")
            for frame_id in self.__bundle_frames:
                    self.__tracks.update(self.db.tracks(frame_id))

            for i, track in enumerate(self.__tracks):

                frames_of_track = self.db.frames(track)
                frames_of_track = [frame for frame in frames_of_track if
                                   self.__first_key_frame_id <= frame <= self.__last_key_frame_id]
                link_loop = self.db.link(frameId=frames_of_track[-1], trackId=track)
                if self.bad_match(link_loop):
                    print("Triangulation of link is bad")

                    print(f"Left keypoint: {link_loop.left_keypoint()}")
                    print(f"Right keypoint: {link_loop.right_keypoint()}")
                    # continue
                p3d_gtsam = triangulate_gtsam(self.__transformations[frames_of_track[-1] - self.__first_key_frame_id],
                                              calib_mat_gtsam, link_loop)
                landmark_gtsam_symbol = symbol(LAND_MARK_SYM, track)
                factors = []
                good_link = True
                for frame_id in frames_of_track:
                    # Factor creation
                    # if frame_id == f
                    if self.bad_match(self.db.link(frame_id, track)):
                        print(f"Link is not good for track {i} in frame {frame_id}")
                        print(f"Left keypoint: {self.db.link(frame_id, track).left_keypoint()}")
                        print(f"Right keypoint: {self.db.link(frame_id, track).right_keypoint()}")
                        good_link = False
                        break
                    frame_l_xy = self.db.link(frame_id, track).left_keypoint()
                    frame_r_xy = self.db.link(frame_id, track).right_keypoint()

                    gtsam_measurement_pt2 = gtsam.StereoPoint2(frame_l_xy[0], frame_r_xy[0], frame_l_xy[1])
                    projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                    camera_symbol_gtsam = symbol("c", frame_id - self.__first_key_frame_id)

                    factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                                         camera_symbol_gtsam, symbol(LAND_MARK_SYM, track),
                                                         calib_mat_gtsam)
                    self.__camera_sym.add(camera_symbol_gtsam)
                    factors.append(factor)

                if not good_link:
                    continue
                self.__initial_estimate.insert(landmark_gtsam_symbol, p3d_gtsam)
                self.__landmark_sym.add(landmark_gtsam_symbol)
                for fac in factors:
                    self.graph.add(fac)


        else:
            print(f"Creating factor graph for frames {self.__first_key_frame_id} and {self.__last_key_frame_id} - Loop-Closure")
            # Create Tracks

            self.__tracks.update(self.__little_bundle_track)

            for i, link_loop in enumerate(self.__tracks):
                # todo refactor the link for macth the trinagulate_gtsam function
                frames_of_track = self.__bundle_frames

                #Todo : Triangulate the point of one of the frames (key frame or prev frame)

                p3d_gtsam = triangulate_gtsam(self.__transformations[0],
                                              calib_mat_gtsam, link_loop, True)
                landmark_gtsam_symbol = symbol(LAND_MARK_SYM, i)
                factors = []

                for frame in range(2):

                    if frame == 1:
                        # measurments of the prev_frame
                        frame_l_xy = link_loop.left_keypoint_prev()
                        frame_r_xy = link_loop.right_keypoint_prev()

                    else:
                        #measurments of the kf
                        frame_l_xy = link_loop.left_keypoint_kf()
                        frame_r_xy = link_loop.right_keypoint_kf()
                    # Todo in which value for y to choose (because we dont in the stereo-regular case
                    gtsam_measurement_pt2 = gtsam.StereoPoint2(frame_l_xy[0], frame_r_xy[0], (frame_l_xy[1] + frame_r_xy[1]) / 2)
                    # todo: make sure that the link_loop sent to the function appropetely - compare it to the db.link in the other case.
                    projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                    camera_symbol_gtsam = symbol("c", frame)
                    factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                                         camera_symbol_gtsam, symbol(LAND_MARK_SYM, i),
                                                         calib_mat_gtsam)
                    self.__camera_sym.add(camera_symbol_gtsam)
                    factors.append(factor)


                self.__initial_estimate.insert(landmark_gtsam_symbol, p3d_gtsam)
                self.__landmark_sym.add(landmark_gtsam_symbol)
                for fac in factors:
                    self.graph.add(fac)

    def bad_match(self, link):
        return link.left_keypoint()[0] < (link.right_keypoint()[0] + EPSILON)

    def compose_transformations(self):

        for i in range(len(self.__bundle_frames)):

            if i == 0:
                current_Rt_gtsam = gtsam.Pose3()
                prev = current_Rt_gtsam

                self.__transformations.append(current_Rt_gtsam)

                camera_symbol_gtsam = symbol(CAMERA_SYM, i)
                self.__initial_estimate.insert(camera_symbol_gtsam, current_Rt_gtsam)
                self.__camera_sym.add(camera_symbol_gtsam)
                sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)  # todo: what about frame[0](0,0,0)

                # Initial pose
                camera_symbol_gtsam = symbol(CAMERA_SYM, 0)
                factor = gtsam.PriorFactorPose3(camera_symbol_gtsam, current_Rt_gtsam, pose_uncertainty)
                self.__camera_sym.add(camera_symbol_gtsam)
                self.graph.add(factor)

            else:
                #todo: avish tried compose differently
                # convert to inverse by gtsam.Pose3.inverse
                current_left_transformation_homogenus = self.compose_to_first_kf(self.__bundle_frames[i])
                Rt_inverse_gtsam = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
                self.__transformations.append(Rt_inverse_gtsam)
                camera_symbol_gtsam = symbol(CAMERA_SYM, i)
                self.__initial_estimate.insert(camera_symbol_gtsam, Rt_inverse_gtsam)

                self.__camera_sym.add(camera_symbol_gtsam)



                #
                # compose_curr_Rt_gtsam = current_Rt_gtsam.compose(prev)
                # prev = compose_curr_Rt_gtsam
                # self.__transformations.append(compose_curr_Rt_gtsam)
                # camera_symbol_gtsam = symbol(CAMERA_SYM, i)
                # self.__initial_estimate.insert(camera_symbol_gtsam, compose_curr_Rt_gtsam)
                # self.__camera_sym.add(camera_symbol_gtsam)
        return



    def calculate_graph_error(self, optimized=True):

        if not optimized:
            error = self.graph.error(self.__initial_estimate)
        else:
            error = self.graph.error(self.__optimized_values)

        return np.log(error)

    def optimize(self):
        """
        Apply optimization with Levenberg marquardt algorithm
        """
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.__initial_estimate)
        self.__optimized_values = self.optimizer.optimize()

    def get_initial_estimate(self):
        return self.__initial_estimate

    def get_optimized_values(self):
        return self.__optimized_values

    def get_cameras_symbols_lst(self):
        """
        Return cameras symbols list
        """
        return self.__camera_sym

    def get_landmarks_symbols_lst(self):
        """
        Returns landmarks symbols list
        """
        return self.__landmark_sym


    def get_optimized_cameras(self):

        cams = []
        for frame_id in range(self.__first_key_frame_id, self.__last_key_frame_id + 1):
            cams.append(self.__optimized_values.atPose3(symbol(CAMERA_SYM, frame_id)))
        return cams


    def get_optimized_last_camera(self, loop=False):
        """
        Get the optimized last camera
        Returns:
            Pose3: The optimized last camera
        """
        if loop:
            return self.__optimized_values.atPose3(symbol(CAMERA_SYM, 1))
        return self.__optimized_values.atPose3(symbol(CAMERA_SYM, self.__last_key_frame_id - self.__first_key_frame_id))

    def get_optimized_first_camera(self):
        """
        Get the optimized first camera
        Returns:
            Pose3: The optimized first camera
        """
        return self.__optimized_values.atPose3(symbol(CAMERA_SYM, 0))

    def get_optimized_landmarks(self):
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmarks.append(self.__optimized_values.atPoint3(landmark_sym))

        return np.asarray(landmarks)


    def get_optimized_landmarks_lst(self):
        """
        Get the optimized landmarks
        Returns:
            list: The optimized landmarks
        """
        landmarks = []
        for landmark_sym in self.__landmark_sym:
            landmark = self.__optimized_values.atPoint3(landmark_sym)
            landmarks.append(landmark)
        return landmarks

    #todo: function that avish added
    def get_homogeneous_transformation(self, frame_id):
        matrix = self.db.rotation_matrices[frame_id]
        translation = self.db.translation_vectors[self.__first_key_frame_id]
        translation = translation.reshape(3, 1)
        R = np.concatenate((matrix, np.zeros((1, 3))), axis=0)
        zero_row = np.array([0]).reshape(1, 1)
        translation_homogenus = np.vstack((translation, zero_row))
        # transformation_homogenus = np.concatenate((R, translation_homogenus.reshape(4, 1)), axis=1)
        return R, translation_homogenus

    def get_frame_range(self):
        return (self.__first_key_frame_id, self.__last_key_frame_id)

    def compose_to_first_kf(self, frame_id):
        kf_R, kf_t = self.get_homogeneous_transformation(self.__first_key_frame_id)
        frame_R, frame_t = self.get_homogeneous_transformation(frame_id)
        compose_R = kf_R.T @ frame_R
        compose_t = kf_R.T @ (frame_t - kf_t)
        compose_Rt = np.concatenate((compose_R, compose_t), axis=1)
        return compose_Rt

    def get_landmarks_symbols_set(self):
        """
        Returns landmarks symbols list
        """
        return self.__landmark_sym

    # def __getstate__(self):
    #     """Exclude unpicklable objects and prepare data for serialization."""
    #     state = self.__dict__.copy()
    #     # Remove unpicklable objects
    #     del state['optimizer']
    #     del state['graph']
    #     del state['_BundleWindow__initial_estimate']
    #     del state['_BundleWindow__optimized_values']
    #
    #     # Prepare data to reconstruct the graph
    #     state['_graph_data'] = self._serialize_graph()
    #
    #     # Prepare data to reconstruct initial estimates
    #     state['_initial_estimate_data'] = self._serialize_values(self.__initial_estimate)
    #
    #     # Prepare data to reconstruct optimized values
    #     if self.__optimized_values is not None:
    #         state['_optimized_values_data'] = self._serialize_values(self.__optimized_values)
    #     else:
    #         state['_optimized_values_data'] = None
    #
    #     return state
    #
    # def __setstate__(self, state):
    #     """Restore state and reconstruct unpicklable objects."""
    #     # Restore the dict excluding the GTSAM objects
    #     self.__dict__.update(state)
    #     # Re-initialize optimizer
    #     self.optimizer = None
    #
    #     # Reconstruct the graph
    #     self.graph = gtsam.NonlinearFactorGraph()
    #     self._deserialize_graph(self._graph_data)
    #
    #     # Reconstruct initial estimates
    #     self.__initial_estimate = gtsam.Values()
    #     self._deserialize_values(self.__initial_estimate_data, self.__initial_estimate)
    #
    #     # Reconstruct optimized values
    #     if self._optimized_values_data is not None:
    #         self.__optimized_values = gtsam.Values()
    #         self._deserialize_values(self._optimized_values_data, self.__optimized_values)
    #     else:
    #         self.__optimized_values = None
    #
    #     # Serialization helpers

    # def _serialize_graph(self):
    #     """Serialize the graph to a picklable format."""
    #     graph_data = []
    #     for factor in self.graph:
    #         # Serialize each factor individually
    #         factor_data = self._serialize_factor(factor)
    #         graph_data.append(factor_data)
    #     return graph_data
    #
    # def _deserialize_graph(self, graph_data):
    #     """Deserialize the graph from the picklable format."""
    #     for factor_data in graph_data:
    #         factor = self._deserialize_factor(factor_data)
    #         self.graph.add(factor)
    #
    # def _serialize_values(self, values):
    #     """Serialize Values object to a picklable format."""
    #     values_data = {}
    #     for key in values.keys():
    #         value = values.at(key)
    #         values_data[key] = self._serialize_value(value)
    #     return values_data
    #
    # def _deserialize_values(self, values_data, values):
    #     """Deserialize Values object from picklable format."""
    #     for key, value_data in values_data.items():
    #         value = self._deserialize_value(value_data)
    #         values.insert(key, value)
    #
    # def _serialize_factor(self, factor):
    #     """Serialize a factor to a picklable format."""
    #     # Implement serialization for different factor types
    #     # For example, for GenericStereoFactor3D
    #     if isinstance(factor, gtsam.GenericStereoFactor3D):
    #         factor_data = {
    #             'type': 'GenericStereoFactor3D',
    #             'measurement': [factor.measured().uL(), factor.measured().uR(), factor.measured().v()],
    #             'noise_model': factor.noiseModel(),
    #             'keys': list(factor.keys()),
    #             'calibration': factor.calibration()
    #         }
    #         # Note: noise_model and calibration may need special handling
    #         return factor_data
    #     elif isinstance(factor, gtsam.PriorFactorPose3):
    #         # Handle PriorFactorPose3
    #         pose = factor.prior()
    #         pose_data = self._serialize_value(pose)
    #         factor_data = {
    #             'type': 'PriorFactorPose3',
    #             'pose': pose_data,
    #             'noise_model': factor.noiseModel(),
    #             'key': factor.keys()[0]
    #         }
    #         # Note: noise_model may need special handling
    #         return factor_data
    #     else:
    #         raise NotImplementedError(f"Serialization for factor type {type(factor)} is not implemented.")
    #
    # def _deserialize_factor(self, factor_data):
    #     """Deserialize a factor from picklable format."""
    #     if factor_data['type'] == 'GenericStereoFactor3D':
    #         measurement = gtsam.StereoPoint2(*factor_data['measurement'])
    #         # Recreate noise model and calibration
    #         noise_model = factor_data['noise_model']  # Handle appropriately
    #         calibration = factor_data['calibration']  # Handle appropriately
    #         factor = gtsam.GenericStereoFactor3D(
    #             measurement, noise_model, factor_data['keys'][0],
    #             factor_data['keys'][1], calibration)
    #         return factor
    #     elif factor_data['type'] == 'PriorFactorPose3':
    #         pose = self._deserialize_value(factor_data['pose'])
    #         noise_model = factor_data['noise_model']  # Handle appropriately
    #         factor = gtsam.PriorFactorPose3(factor_data['key'], pose, noise_model)
    #         return factor
    #     else:
    #         raise NotImplementedError(f"Deserialization for factor type {factor_data['type']} is not implemented.")
    #
    # def _serialize_value(self, value):
    #     """Serialize a GTSAM value to a picklable format."""
    #     if isinstance(value, gtsam.Pose3):
    #         rotation = value.rotation().matrix()
    #         translation = value.translation()
    #         value_data = {
    #             'type': 'Pose3',
    #             'rotation': rotation.tolist(),
    #             'translation': translation.tolist()
    #         }
    #         return value_data
    #     elif isinstance(value, gtsam.Point3):
    #         value_data = {
    #             'type': 'Point3',
    #             'point': value.tolist()
    #         }
    #         return value_data
    #     else:
    #         raise NotImplementedError(f"Serialization for value type {type(value)} is not implemented.")
    #
    # def _deserialize_value(self, value_data):
    #     """Deserialize a GTSAM value from picklable format."""
    #     if value_data['type'] == 'Pose3':
    #         rotation = np.array(value_data['rotation'])
    #         translation = np.array(value_data['translation'])
    #         pose = gtsam.Pose3(gtsam.Rot3(rotation), translation)
    #         return pose
    #     elif value_data['type'] == 'Point3':
    #         point = np.array(value_data['point'])
    #         return gtsam.Point3(point)
    #     else:
    #         raise NotImplementedError(f"Deserialization for value type {value_data['type']} is not implemented.")


def solve_bundle_window(db, first_frame, last_frame):
    window = BundleWindow(db, first_frame, last_frame)
    window.create_factor_graph()
    error_before_optim = window.calculate_graph_error(False)
    window.optimize()
    error_after_optim = window.calculate_graph_error()
    return window, error_before_optim, error_after_optim
