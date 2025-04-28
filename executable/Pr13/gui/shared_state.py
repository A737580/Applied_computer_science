class SharedState:
    def __init__(self):
        # --- Переменные состояния ---
        self.root_dir = None
        self.selected_genre = None
        self.current_img_size = (128, 128)
        self.current_test_size = 0.25
        self.current_random_state = 42
        self.inference_dir = None
        self.inference_image_files = []
        self.current_inference_image_index = -1
        self.class_names = []
        self.class_to_idx = {}
        self.last_run_results = {}
        self.trained_model = None
        self.loaded_inference_model_path = None
        self.loaded_inference_model = None
        self.loaded_inference_class_names = None
        