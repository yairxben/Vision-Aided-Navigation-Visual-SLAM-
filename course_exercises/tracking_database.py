import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer

NO_ID = -1


""" a class that holds a single keypoint data for both left and right images of a stereo frame """
class Link:
    x_left: float
    x_right: float
    y: float

    def __init__(self, x_left, x_right, y):
        self.x_left = x_left
        self.x_right = x_right
        self.y = y

    def left_keypoint(self):
        return np.array([self.x_left, self.y])

    def right_keypoint(self):
        return np.array([self.x_right, self.y])

    def __str__(self):
        return f'Link (xl={self.x_left}, xr={self.x_right}, y={self.y})'


"""
for internal use of TrackingDB.
holds the internal data of a single match that is needed for the tracking process. """
class MatchLocation:
    distance: float
    loc: int
    is_new_track: bool
    MAX_DIST = float('Inf')

    # distance - the quality of the match
    # loc - the index of the feature matched in the query descriptor array (the row of the feature)
    # is_new_track - True iff this is the first match of that TrackId (a new track has been established)
    def __init__(self, distance=None, loc=None, is_new_track=None):
        if distance is None:
            distance = MatchLocation.MAX_DIST
        self.distance = distance
        self.loc = loc
        self.is_new_track = is_new_track

    def valid(self):
        return self.distance < self.MAX_DIST and self.loc is not None and self.is_new_track is not None

    def __lt__(self, distance):
        return self.distance < distance

    def __le__(self, distance):
        return self.distance <= distance

    def __gt__(self, distance):
        return self.distance > distance

    def __ge__(self, distance):
        return self.distance >= distance

    def __str__(self):
        return f'MatchLocation (dist={self.distance}, loc={self.loc}, is_new={self.is_new_track})'


"""
A database to accumulate the tracking information between all consecutive video frames.
The stereo frames are added sequentially with their tracking information and arranged
such that the data can be referenced using Ids for the frames and tracks.
"""
class TrackingDB:
    last_frameId: int
    last_trackId: int
    trackId_to_frames: Dict[int, List[int]]
    linkId_to_link: Dict[Tuple[int, int], Link]
    frameId_to_lfeature: Dict[int, np.ndarray]
    frameId_to_trackIds_list: Dict[int, List[int]]
    prev_frame_links: List[Link]
    leftover_links: Dict[int, List[Link]]

    def __init__(self):
        self.last_frameId = -1  # assumptions: frameIds are consecutive from 0 (1st frame) to last_frameId
        self.last_trackId = -1
        self.trackId_to_frames = {}  # map trackId --> frameId list   // all the frames the track appears on
        self.linkId_to_link = {}  # map (frameId, trackId) --> Link
        self.frameId_to_lfeature = {}  # map frameId --> np.ndarray       // the descriptors array
        # trackId for every line in the descriptor array of that frame:
        self.frameId_to_trackIds_list = {}  # map frameId --> trackId list
        # the links associated with the previous added frame.
        # will be added to linkId_to_link if they are matched to the next frame:
        self.prev_frame_links = None
        # map frameId --> link list, all the links of features that were not matched
        # ordered according to the order in the descriptors array
        self.leftover_links = {}

    """ a list of the frames on trackId """
    def frames(self, trackId) -> List[int]:
        return self.trackId_to_frames.get(trackId, [])

    """ all links that are part of the trackId. returns a dict frameId --> Link """
    def track(self, trackId) -> Dict[int, Link]:
        fIds = self.frames(trackId)
        track_links = {}
        for fId in fIds:
            track_links[fId] = self.linkId_to_link[(fId, trackId)]
        return track_links

    """ the last frame of trackId """
    def last_frame_of_track(self, trackId) -> int:
        return self.frames(trackId)[-1]

    """ a list of the tracks on frameId """
    def tracks(self, frameId) -> List[int]:
        tracks = self.frameId_to_trackIds_list.get(frameId, None)
        if not tracks:
            return []
        return sorted([x for x in tracks if x != NO_ID])

    """ number of tracks issued """
    def track_num(self) -> int:
        return len(self.all_tracks())

    """ all valid trackIds """
    def all_tracks(self) -> List[int]:
        return list(self.trackId_to_frames.keys())

    """ total number of links in the DB """
    def link_num(self) -> int:
        return len(self.linkId_to_link)

    """ number of frames issued """
    def frame_num(self) -> int:
        return self.last_frameId + 1

    """ a range of all the frames """
    def all_frames(self) -> Sequence[int]:
        return range(self.frame_num())

    """ The feature array of (the left image of) frameId """
    def features(self, frameId) -> Optional[np.ndarray]:
        return self.frameId_to_lfeature.get(frameId, None)

    """ The feature array of (the left image of) the last added frame """
    def last_features(self) -> Optional[np.ndarray]:
        return self.frameId_to_lfeature.get(self.last_frameId, None)

    """ the link of trackId that sits on frameId """
    def link(self, frameId, trackId) -> Optional[Link]:
        return self.linkId_to_link.get((frameId, trackId), None)

    """ all links that are part of a track on frameId. returns a dict trackId --> Link """
    def links(self, frameId) -> Dict[int, Link]:
        frame_links = {}
        for key, link in self.linkId_to_link.items():
            if key[0] == frameId:
                frame_links[key[1]] = link
        return frame_links

    """ all the links of the last frame,
        not only the ones that are part of a track but every extracted feature """
    def all_last_frame_links(self) -> List[Link]:
        return self.prev_frame_links

    """ all the links of frameId,
        not only the ones that are part of a track but every extracted feature from the frame """
    def all_frame_links(self, frameId) -> List[Link]:
        feat_num = self.frameId_to_lfeature[frameId].shape[0]
        trackIds_list = self.frameId_to_trackIds_list[frameId]
        assert feat_num == len(trackIds_list)

        if frameId == self.last_frameId:
            return self.prev_frame_links
        else:  # build links_for_feat from links in self.linkId_to_link and self.leftover_links
            assert len(self.tracks(frameId)) + len(self.leftover_links[frameId]) == feat_num
            leftover_i = 0
            links_for_feat = [Link(0, 0, 0)] * feat_num
            for i, trackId in enumerate(trackIds_list):
                if trackId != NO_ID:
                    links_for_feat[i] = self.link(frameId, trackId)
                else:
                    links_for_feat[i] = self.leftover_links[frameId][leftover_i]
                    leftover_i += 1
        return links_for_feat

    """ issue a new frame and return its frameId """
    def issue_frameId(self) -> int:
        self.last_frameId += 1
        return self.last_frameId

    """ issue a new track and return its trackId """
    def issue_trackId(self) -> int:
        self.last_trackId += 1
        return self.last_trackId

    """
    Processes the output of a opencv match/knnMatch between the left and right images of
    a stereo frame into structures viable for trackingDB.
 
    Output:
    features: np.ndarray. The feature array, same as the input features but with only the features
              that have a valid match (feature row is removed if it has no valid match)
    links: List[Link]. The links associated with these frame features.
 
    Input:
    features: np.ndarray. The feature array, same format as opencv.
              The features of the left image of the frame.
              Supplied in order to remove the outliers from the matrix
    kp_left: Tuple[cv2.KeyPoint]. Keypoints of the left image of the stereo frame as supplied by opencv 'detect'.
             Should match the features number of features in 'features' matrix.
             i.e. features.shape[0] == len(kp_left)
    kp_right: Tuple[cv2.KeyPoint]. Keypoints of the right image of the stereo frame as supplied by opencv 'detect'.
    matches: Tuple[cv2.DMatch], Can also be Tuple[Tuple[cv2.DMatch]] if using knnMatch.
             Same format as opencv. Holds matches from left to right image of the frame.
             - in the call to opencv Match/knnMatch use Match(leftDescriptors, rightDescriptors)
    inliers: List[bool], optional. A list of booleans indicating if the matches are inliers.
             (inliers[i] indicates the validity of matches[i]), i.e. len(inliers) == len(matches)
             If omitted treats all the matches as inliers.
    """
    @staticmethod
    def create_links(features: np.ndarray,
                     kp_left: Tuple[cv2.KeyPoint],
                     kp_right: Tuple[cv2.KeyPoint],
                     matches: Tuple[cv2.DMatch],
                     inliers: List[bool] = None) -> Tuple[np.ndarray, List[Link]]:
        assert features.shape[0] == len(kp_left)
        is_knn = type(matches[0]) is tuple
        inliers = TrackingDB.__all_inliers(inliers, len(matches))
        links = []
        is_valid = [False] * len(kp_left)
        for m, inlier in zip(matches, inliers):
            if not inlier:
                continue
            m = m[0] if is_knn else m
            is_valid[m.queryIdx] = True
            kpl = kp_left[m.queryIdx]
            kpr = kp_right[m.trainIdx]

            link = Link(kpl.pt[0], kpr.pt[0], (kpl.pt[1] + kpr.pt[1]) / 2)
            links.append(link)

        return features[is_valid], links

    """
    Adds a new stereo frame including all its information and the
    matches to the previous frame to the tracking database.
 
    Output: The frameId assigned to the new frame.
 
    Input:
    links: List[Link]. The links associated with this frame features.
           holds the information of the left and right keypoints of each feature.
           Should match the left_features matrix by position (link[i] should match feature at line i)
           i.e. left_features.shape[0] == len(links)
    left_features: np.ndarray. The feature array, same format as opencv.
                   Holds the features of the left image of the frame.
    matches_to_previous_left: Tuple[cv2.DMatch], optional. Can also be Tuple[Tuple[cv2.DMatch]] if using knnMatch.
                              Can also work with List[cv2.DMatch] / List[Tuple[cv2.DMatch]].
                              Holds matches from the previous frame to the current frame.
                              Same format as opencv. When matching the previous frame left image should be the
                              query image and the current frame left image should be the train image.
                              - in the call to opencv Match/knnMatch use Match(queryDescriptors, trainDescriptors)
                              in the call for the first frame leave matches_to_previous_left None.
                              Should have the same length as the previous frame feature number.
    inliers: List[bool], optional. A list of booleans indicating if the matches are inliers.
             (inliers[i] indicates the validity of matches_to_previous_left[i]),
             i.e. len(inliers) == len(matches_to_previous_left). If omitted treats all the matches as inliers.
    """
    def add_frame(self,
                  links: List[Link],
                  left_features: np.ndarray,
                  matches_to_previous_left: Tuple[cv2.DMatch] = None,
                  inliers: List[bool] = None) -> int:
        feat_num = left_features.shape[0]
        assert feat_num == len(links)

        prev_frameId = self.last_frameId
        cur_frameId = self.issue_frameId()
        self.frameId_to_lfeature[cur_frameId] = left_features
        self.frameId_to_trackIds_list[cur_frameId] = [NO_ID] * len(links)
        if cur_frameId == 0:  # first frame
            self.prev_frame_links = links
            assert matches_to_previous_left is None
            return cur_frameId

        assert matches_to_previous_left is not None  # should have matches to prev frame (unless first frame)
        inliers = self.__all_inliers(inliers, len(matches_to_previous_left))
        assert self.frameId_to_lfeature[prev_frameId].shape[0] == len(matches_to_previous_left) == len(inliers)

        # get prev frame trackIds:
        prev_frame_tracksIds = self.frameId_to_trackIds_list.get(prev_frameId)
        assert prev_frame_tracksIds is not None

        # go over all matches to previous frame:
        is_knn = type(matches_to_previous_left[0]) is tuple
        prev_matches = [MatchLocation()] * feat_num
        for m, inlier in zip(matches_to_previous_left, inliers):
            if not inlier:
                continue
            m = m[0] if is_knn else m
            prev_feat_loc = m.queryIdx
            cur_feat_loc = m.trainIdx
            prev_match = prev_matches[cur_feat_loc]
            if prev_match <= m.distance:  # a better kp was already matched to that kp, skip
                continue
            if prev_match.valid():  # found a better match, erase previous match
                prev_trackId = prev_frame_tracksIds[prev_match.loc]
                assert prev_trackId == self.frameId_to_trackIds_list[cur_frameId][cur_feat_loc]
                if prev_match.is_new_track:
                    prev_frame_tracksIds[prev_match.loc] = NO_ID  # reset to no track
                    del self.linkId_to_link[(prev_frameId, prev_trackId)]  # remove link of wrong match
                    removed_frameId = self.trackId_to_frames[prev_trackId].pop(0)  # remove 1st frame from track list
                    assert removed_frameId == prev_frameId
                # regardless if new or not, remove link and track from current frame:
                self.__remove_link_from_last_frame(prev_trackId, cur_feat_loc)  # remove 2nd link of wrong match

            is_new_track = prev_frame_tracksIds[prev_feat_loc] == NO_ID  # 1st match
            prev_matches[cur_feat_loc] = MatchLocation(m.distance, prev_feat_loc, is_new_track)
            if is_new_track:
                new_trackId = self.issue_trackId()
                self.__new_link(prev_frameId, new_trackId, prev_feat_loc, self.prev_frame_links[prev_feat_loc])
                assert prev_frame_tracksIds[prev_feat_loc] == new_trackId

            self.__new_link(cur_frameId, prev_frame_tracksIds[prev_feat_loc], cur_feat_loc, links[cur_feat_loc])

        # store all links of features in previous frame that were not matched:
        self.leftover_links[prev_frameId] = []
        for link, trackId in zip(self.prev_frame_links, prev_frame_tracksIds):
            if trackId == NO_ID:
                self.leftover_links[prev_frameId].append(link)

        self.prev_frame_links = links
        return cur_frameId

    """ save TrackingDB to base_filename+'.pkl' file. """
    def serialize(self, base_filename):
        data = {
            'last_frameId': self.last_frameId,
            'last_trackId': self.last_trackId,
            'trackId_to_frames': self.trackId_to_frames,
            'linkId_to_link': self.linkId_to_link,
            'frameId_to_lfeature': self.frameId_to_lfeature,
            'frameId_to_trackIds_list': self.frameId_to_trackIds_list,
            'prev_frame_links': self.prev_frame_links,
            'leftover_links': self.leftover_links
        }
        filename = base_filename + '.pkl'
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print('TrackingDB serialized to', filename)

    """ load TrackingDB to base_filename+'.pkl' file. """
    def load(self, base_filename):
        filename = base_filename + '.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.last_frameId = data['last_frameId']
            self.last_trackId = data['last_trackId']
            self.trackId_to_frames = data['trackId_to_frames']
            self.linkId_to_link = data['linkId_to_link']
            self.frameId_to_lfeature = data['frameId_to_lfeature']
            self.frameId_to_trackIds_list = data['frameId_to_trackIds_list']
            self.prev_frame_links = data['prev_frame_links']
            self.leftover_links = data['leftover_links']
        print('TrackingDB loaded from', filename)

    """
    save the data of a single frame to base_filename+'_frameId.pkl' file.  (frameId in six digits with leading zeros)
    serializing the frame holds just the context of the frame without the data needed for continues tracking.
    loading the file will only retrieve the frame data and not update the TrackingDB that holds this frame.
    """
    def serialize_frame(self, base_filename: str, frameId: int):
        data = {
            'frameId': frameId,
            'frame_links': self.all_frame_links(frameId),
            'lfeature': self.frameId_to_lfeature[frameId],
        }
        filename = base_filename + '_{:06d}.pkl'.format(frameId)
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print('TrackingDB frame #', frameId, 'serialized to', filename)

    """
    load a single frame data from base_filename+'_frameId.pkl' file.  (frameId in six digits with leading zeros)
    serializing the frame holds just the context of the frame without the data needed for continues tracking.
    loading the file will only retrieve the frame data and not update the TrackingDB that holds this frame.
    """
    @staticmethod
    def load_frame(base_filename: str, frameId: int) -> Tuple[np.ndarray, List[Link]]:
        filename = base_filename + '_{:06d}.pkl'.format(frameId)
        with open(filename, 'rb') as file:
            data = pickle.load(file)

            saved_frameId = data['frameId']
            assert saved_frameId == frameId

            frame_links = data['frame_links']
            features = data['lfeature']
            print('TrackingDB frame #', frameId, 'loaded from', filename)
            return features, frame_links

    def __add_frameId_to_track(self, frameId, trackId):
        if trackId in self.trackId_to_frames:
            self.trackId_to_frames[trackId].append(frameId)
        else:
            self.trackId_to_frames[trackId] = [frameId]

    def __new_link(self, frameId, trackId, feature_loc, link):
        assert frameId == self.last_frameId or self.frameId_to_trackIds_list[frameId][feature_loc] == NO_ID
        self.frameId_to_trackIds_list[frameId][feature_loc] = trackId
        assert (frameId, trackId) not in self.linkId_to_link
        self.linkId_to_link[(frameId, trackId)] = link
        self.__add_frameId_to_track(frameId, trackId)

    @staticmethod
    def __all_inliers(inliers, n):
        if inliers is None:
            return [True] * n
        assert len(inliers) == n
        return inliers

    def __remove_link_from_last_frame(self, trackId, feat_loc_on_trackId_list):
        del self.linkId_to_link[(self.last_frameId, trackId)]
        self.frameId_to_trackIds_list[self.last_frameId][feat_loc_on_trackId_list] = NO_ID
        assert self.trackId_to_frames[trackId][-1] == self.last_frameId and 'last track frame is not the last frame'
        self.__remove_last_frame_from_track_list(trackId)

    def __remove_last_frame_from_track_list(self, trackId):
        removed_frameId = self.trackId_to_frames[trackId].pop()  # remove frame from track list
        assert removed_frameId == self.last_frameId
        if not self.trackId_to_frames[trackId]:  # if empty remove list
            del self.trackId_to_frames[trackId]

    def _check_consistency(self):
        n = self.link_num()

        start = timer()
        link_count = 0
        for fId in self.all_frames():
            frame_links_num = len(self.links(fId))
            assert frame_links_num == len(self.tracks(fId))
            link_count += frame_links_num
            print(fId, ':  +', frame_links_num, '=', link_count, '/', n)
        assert link_count == n
        print('Elapsed time: {0:.2f} secs.'.format(timer() - start))

        start = timer()
        link_count = 0
        for tId in self.all_tracks():
            track_frames = self.frames(tId)
            track_len = len(track_frames)
            assert track_len >= 2
            link_count += track_len
        assert link_count == n
        print('Elapsed time: {0:.2f} secs.'.format(timer() - start))

        start = timer()
        for (frameId, trackId), link in self.linkId_to_link.items():
            assert frameId in self.frames(trackId)
            assert trackId in self.tracks(frameId)

        print('Elapsed time: {0:.2f} secs.'.format(timer() - start))
        print('All Good')
