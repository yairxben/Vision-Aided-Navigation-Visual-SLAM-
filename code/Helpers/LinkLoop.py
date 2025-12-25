import numpy as np



class LinkLoop:
    link_loop_id: int
    prev_frame_id: float
    key_frame_id : float
    x_prev_frame_left: float
    y_prev_frame_left: float
    x_key_frame_left: float
    y_key_frame_left: float
    x_prev_frame_right: float
    y_prev_frame_right: float
    x_key_frame_right: float
    y_key_frame_right: float

    def __init__(self, link_loop_id: int, prev_frame_id: float, key_frame_id: float,
                 x_prev_frame_left: float, y_prev_frame_left: float,
                 x_key_frame_left: float, y_key_frame_left: float,
                 x_prev_frame_right: float, y_prev_frame_right: float,
                 x_key_frame_right: float, y_key_frame_right: float):
        self.prev_frame_id = prev_frame_id
        self.key_frame_id = key_frame_id
        self.x_prev_frame_left = x_prev_frame_left
        self.y_prev_frame_left = y_prev_frame_left
        self.x_key_frame_left = x_key_frame_left
        self.y_key_frame_left = y_key_frame_left
        self.x_prev_frame_right = x_prev_frame_right
        self.y_prev_frame_right = y_prev_frame_right
        self.x_key_frame_right = x_key_frame_right
        self.y_key_frame_right = y_key_frame_right




    def get_link_loop_id(self):
        return self.link_loop_id

    # Implement getters
    def left_keypoint_prev(self):
        return np.array([self.x_prev_frame_left, self.y_prev_frame_left])


    def right_keypoint_prev(self):
        return np.array([self.x_prev_frame_right, self.y_prev_frame_right])


    def left_keypoint_kf(self):
        return np.array([self.x_key_frame_left, self.y_key_frame_left])


    def right_keypoint_kf(self):
        return np.array([self.x_key_frame_right, self.y_key_frame_right])



    def get_prev_frame_id(self):
        return self.prev_frame_id

    def get_key_frame_id(self):
        return self.key_frame_id

    def get_x_prev_frame_left(self):
        return self.x_prev_frame_left

    def get_y_prev_frame_left(self):
        return self.y_prev_frame_left

    def get_x_key_frame_left(self):
        return self.x_key_frame_left

    def get_y_key_frame_left(self):
        return self.y_key_frame_left


    def get_x_prev_frame_right(self):
        return self.x_prev_frame_right

    def get_y_prev_frame_right(self):
        return self.y_prev_frame_right

    def get_x_key_frame_right(self):
        return self.x_key_frame_right

    def get_y_key_frame_right(self):
        return self.y_key_frame_right