from customtkinter import *
from capture_devices import devices
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import cv2
import pickle
from PIL import Image
import numpy as np
import os
from collections import Counter
import shutil

global img_path
face_pics = os.listdir('./data/Train_faces')

camera_list = [n.replace('DEVICE NAME : ', '') for n in devices.run_with_param(device_type='video', result_=True)]

facedetect = cv2.CascadeClassifier('framework/haarcascade_frontalface_default.xml')
app = CTk()

app.bind('<Escape>', lambda e: app.quit())

# Global variables for current model
current_model = None
current_model_type = "knn"


hyperparams = {
    "knn": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto"},
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42},
    "rf": {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": -1}
}



def show_dataset_stats():
    """Display statistics about the dataset: number of samples per person"""
    if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
        error_window = CTkToplevel(app)
        error_window.title("Error")
        error_window.geometry("300x100")
        CTkLabel(error_window, text="No training data found!").pack(pady=20)
        CTkButton(error_window, text="OK", command=error_window.destroy).pack()
        return

    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Count samples per person
    name_counts = Counter(LABELS)

    # Create statistics window
    stats_window = CTkToplevel(app)
    stats_window.title("Dataset Statistics")
    stats_window.geometry("600x500")

    # Header
    header_frame = CTkFrame(stats_window)
    header_frame.pack(fill="x", padx=10, pady=10)
    CTkLabel(header_frame, text="DATASET STATISTICS", font=("Arial", 18, "bold")).pack()

    # Summary
    summary_frame = CTkFrame(stats_window)
    summary_frame.pack(fill="x", padx=10, pady=5)
    CTkLabel(summary_frame, text=f"Total People: {len(name_counts)}", font=("Arial", 14)).pack(anchor="w", padx=10)
    CTkLabel(summary_frame, text=f"Total Face Samples: {len(LABELS)}", font=("Arial", 14)).pack(anchor="w", padx=10)

    # Scrollable frame for list of people
    scroll_frame = CTkScrollableFrame(stats_window, height=250)
    scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

    CTkLabel(scroll_frame, text="Samples per person:", font=("Arial", 14, "bold")).pack(anchor="w", pady=5)

    # Display each person with their sample count and delete button
    for person, count in sorted(name_counts.items()):
        person_frame = CTkFrame(scroll_frame)
        person_frame.pack(fill="x", padx=5, pady=2)

        CTkLabel(person_frame, text=f"• {person}:", font=("Arial", 12), width=120, anchor="w").pack(side="left", padx=5)
        CTkLabel(person_frame, text=f"{count} samples", font=("Arial", 12), text_color="green").pack(side="left",
                                                                                                     padx=5)

        # Visual progress bar
        progress_bar = CTkProgressBar(person_frame, width=150, height=10)
        progress_bar.pack(side="left", padx=10)
        progress_bar.set(min(count / 100, 1.0))

        # Delete button for each person
        delete_btn = CTkButton(person_frame, text="Delete", width=60, height=25,
                               fg_color="red", hover_color="darkred",
                               command=lambda p=person: delete_person(p, stats_window))
        delete_btn.pack(side="right", padx=5)

    # Close button
    CTkButton(stats_window, text="Close", command=stats_window.destroy).pack(pady=10)


def delete_person(person_name, stats_window):
    """Delete a person from the dataset"""
    # Confirm deletion
    confirm_window = CTkToplevel(app)
    confirm_window.title("Confirm Deletion")
    confirm_window.geometry("350x150")
    CTkLabel(confirm_window, text=f"Are you sure you want to delete '{person_name}'?",
             font=("Arial", 12)).pack(pady=20)
    CTkLabel(confirm_window, text=f"This will remove all {person_name}'s samples!",
             text_color="red").pack()

    def confirm_delete():
        # Load current data
        with open('data/names.pkl', 'rb') as w:
            LABELS = pickle.load(w)
        with open('data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)

        # Find indices to keep
        indices_to_keep = [i for i, name in enumerate(LABELS) if name != person_name]

        # Filter data
        new_labels = [LABELS[i] for i in indices_to_keep]
        new_faces = FACES[indices_to_keep]

        # Save filtered data
        with open('data/names.pkl', 'wb') as w:
            pickle.dump(new_labels, w)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(new_faces, f)

        # Delete person's folder from Train_faces if exists
        person_folder = f'./data/Train_faces/{person_name}'
        if os.path.exists(person_folder):
            shutil.rmtree(person_folder)

        # Reset current model
        global current_model
        current_model = None

        # Close confirmation window
        confirm_window.destroy()

        # Close stats window and reopen to refresh
        stats_window.destroy()
        show_dataset_stats()

        # Show success message
        success_window = CTkToplevel(app)
        success_window.title("Success")
        success_window.geometry("250x100")
        CTkLabel(success_window, text=f"{person_name} has been deleted!").pack(pady=20)
        CTkButton(success_window, text="OK", command=success_window.destroy).pack()

    CTkButton(confirm_window, text="Yes, Delete", command=confirm_delete,
              fg_color="red", hover_color="darkred").pack(side="left", padx=20, pady=10)
    CTkButton(confirm_window, text="Cancel", command=confirm_window.destroy,
              fg_color="gray").pack(side="right", padx=20, pady=10)


def load_and_train_model(model_type):
    global current_model, current_model_type
    # Load data
    if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
        return None
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    if len(np.unique(LABELS)) < 2:
        # Not enough classes to train properly
        return None

    # Используем глобальные гиперпараметры
    if model_type == "knn":
        model = KNeighborsClassifier(**hyperparams["knn"])
    elif model_type == "svm":
        model = SVC(**hyperparams["svm"])
    elif model_type == "rf":
        model = RandomForestClassifier(**hyperparams["rf"])
    else:
        model = KNeighborsClassifier(**hyperparams["knn"])

    model.fit(FACES, LABELS)
    current_model = model
    current_model_type = model_type
    return model


def get_current_model():
    global current_model, current_model_type
    if current_model is None or current_model_type != model_choice.get():
        current_model = load_and_train_model(model_choice.get())
    return current_model


def compare_models():
    if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
        error_window = CTkToplevel(app)
        error_window.title("Error")
        error_window.geometry("300x100")
        CTkLabel(error_window, text="No training data found!").pack(pady=20)
        CTkButton(error_window, text="OK", command=error_window.destroy).pack()
        return

    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Check for at least 2 classes
    unique_classes = np.unique(LABELS)
    if len(unique_classes) < 2:
        error_window = CTkToplevel(app)
        error_window.title("Error")
        error_window.geometry("300x100")
        CTkLabel(error_window, text="Need at least 2 persons to compare models!").pack(pady=20)
        CTkButton(error_window, text="OK", command=error_window.destroy).pack()
        return

    # Split data
    if len(unique_classes) > 1:
        X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=0.2, random_state=42,
                                                            stratify=LABELS)
    else:
        X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=0.2, random_state=42)

    results = {}

    # Используем текущие гиперпараметры из глобальной конфигурации
    models = {
        "KNN": KNeighborsClassifier(**hyperparams["knn"]),
        "SVM": SVC(**hyperparams["svm"]),
        "Random Forest": RandomForestClassifier(**hyperparams["rf"])
    }

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        results[name] = f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}"

    # Show results in a new window
    compare_window = CTkToplevel(app)
    compare_window.title("Model Comparison")
    compare_window.geometry("750x500")
    CTkLabel(compare_window, text="Model Performance Comparison", font=("Arial", 16, "bold")).pack(pady=10)

    # Отображаем также использованные гиперпараметры
    params_frame = CTkFrame(compare_window)
    params_frame.pack(fill="x", padx=10, pady=5)
    CTkLabel(params_frame, text="Hyperparameters used:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)

    # KNN params
    knn_frame = CTkFrame(params_frame)
    knn_frame.pack(fill="x", padx=10, pady=2)
    CTkLabel(knn_frame, text=f"KNN: {hyperparams['knn']}", font=("Arial", 10), wraplength=650, justify="left").pack(
        anchor="w")

    # SVM params
    svm_frame = CTkFrame(params_frame)
    svm_frame.pack(fill="x", padx=10, pady=2)
    CTkLabel(svm_frame, text=f"SVM: {hyperparams['svm']}", font=("Arial", 10), wraplength=650, justify="left").pack(
        anchor="w")

    # RF params
    rf_frame = CTkFrame(params_frame)
    rf_frame.pack(fill="x", padx=10, pady=2)
    CTkLabel(rf_frame, text=f"RF: {hyperparams['rf']}", font=("Arial", 10), wraplength=650, justify="left").pack(
        anchor="w")

    CTkLabel(compare_window, text="\nResults:", font=("Arial", 14, "bold")).pack(pady=5)

    for name, res in results.items():
        result_frame = CTkFrame(compare_window)
        result_frame.pack(fill="x", padx=10, pady=5)
        CTkLabel(result_frame, text=f"{name}:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
        CTkLabel(result_frame, text=f"  {res}", font=("Arial", 11)).pack(anchor="w", padx=20)

    CTkButton(compare_window, text="Close", command=compare_window.destroy, width=200).pack(pady=15)


def open_hyperparam_window():
    """Open window to adjust hyperparameters"""
    param_window = CTkToplevel(app)
    param_window.title("Hyperparameter Configuration")
    param_window.geometry("600x700")

    # Tab view for different models
    tabview = CTkTabview(param_window)
    tabview.pack(fill="both", expand=True, padx=10, pady=10)

    # KNN Tab
    knn_tab = tabview.add("KNN")
    CTkLabel(knn_tab, text="K-Nearest Neighbors Parameters", font=("Arial", 14, "bold")).pack(pady=10)

    CTkLabel(knn_tab, text="n_neighbors (1-20):").pack(anchor="w", padx=10)
    knn_n = CTkEntry(knn_tab)
    knn_n.insert(0, str(hyperparams["knn"]["n_neighbors"]))
    knn_n.pack(fill="x", padx=10, pady=5)

    CTkLabel(knn_tab, text="weights (uniform/distance):").pack(anchor="w", padx=10)
    knn_weights = CTkComboBox(knn_tab, values=["uniform", "distance"])
    knn_weights.set(hyperparams["knn"]["weights"])
    knn_weights.pack(fill="x", padx=10, pady=5)

    CTkLabel(knn_tab, text="algorithm (auto/ball_tree/kd_tree/brute):").pack(anchor="w", padx=10)
    knn_algo = CTkComboBox(knn_tab, values=["auto", "ball_tree", "kd_tree", "brute"])
    knn_algo.set(hyperparams["knn"]["algorithm"])
    knn_algo.pack(fill="x", padx=10, pady=5)

    # SVM Tab
    svm_tab = tabview.add("SVM")
    CTkLabel(svm_tab, text="Support Vector Machine Parameters", font=("Arial", 14, "bold")).pack(pady=10)

    CTkLabel(svm_tab, text="C (regularization, 0.1-100):").pack(anchor="w", padx=10)
    svm_c = CTkEntry(svm_tab)
    svm_c.insert(0, str(hyperparams["svm"]["C"]))
    svm_c.pack(fill="x", padx=10, pady=5)

    CTkLabel(svm_tab, text="Kernel (linear, rbf, poly, sigmoid):").pack(anchor="w", padx=10)
    svm_kernel = CTkComboBox(svm_tab, values=["linear", "rbf", "poly", "sigmoid"])
    svm_kernel.set(hyperparams["svm"]["kernel"])
    svm_kernel.pack(fill="x", padx=10, pady=5)

    CTkLabel(svm_tab, text="gamma (scale, auto, or number):").pack(anchor="w", padx=10)
    svm_gamma = CTkEntry(svm_tab)
    svm_gamma.insert(0, str(hyperparams["svm"]["gamma"]))
    svm_gamma.pack(fill="x", padx=10, pady=5)

    # Random Forest Tab
    rf_tab = tabview.add("Random Forest")
    CTkLabel(rf_tab, text="Random Forest Parameters", font=("Arial", 14, "bold")).pack(pady=10)

    CTkLabel(rf_tab, text="n_estimators (10-500):").pack(anchor="w", padx=10)
    rf_n = CTkEntry(rf_tab)
    rf_n.insert(0, str(hyperparams["rf"]["n_estimators"]))
    rf_n.pack(fill="x", padx=10, pady=5)

    CTkLabel(rf_tab, text="max_depth (None or number):").pack(anchor="w", padx=10)
    rf_depth = CTkEntry(rf_tab)
    rf_depth.insert(0, str(hyperparams["rf"]["max_depth"]) if hyperparams["rf"]["max_depth"] else "None")
    rf_depth.pack(fill="x", padx=10, pady=5)

    def save_hyperparams():
        try:
            # Update KNN
            knn_n_val = int(knn_n.get())
            hyperparams["knn"]["n_neighbors"] = max(1, min(20, knn_n_val))
            hyperparams["knn"]["weights"] = knn_weights.get()
            hyperparams["knn"]["algorithm"] = knn_algo.get()

            # Update SVM
            hyperparams["svm"]["C"] = float(svm_c.get())
            hyperparams["svm"]["kernel"] = svm_kernel.get()
            gamma_val = svm_gamma.get()
            if gamma_val.replace('.', '', 1).isdigit():
                hyperparams["svm"]["gamma"] = float(gamma_val)
            else:
                hyperparams["svm"]["gamma"] = gamma_val

            # Update Random Forest
            hyperparams["rf"]["n_estimators"] = int(rf_n.get())
            depth_val = rf_depth.get()
            hyperparams["rf"]["max_depth"] = None if depth_val.lower() == "none" else int(depth_val)

            # Reset current model to force retraining
            global current_model
            current_model = None

            param_window.destroy()

            # Show success message
            success_window = CTkToplevel(app)
            success_window.title("Success")
            success_window.geometry("350x120")
            CTkLabel(success_window, text="Hyperparameters updated successfully!", font=("Arial", 12)).pack(pady=15)
            CTkLabel(success_window, text="Model will retrain on next use.", font=("Arial", 10)).pack()
            CTkButton(success_window, text="OK", command=success_window.destroy, width=150).pack(pady=10)

        except Exception as e:
            error_window = CTkToplevel(app)
            error_window.title("Error")
            error_window.geometry("350x120")
            CTkLabel(error_window, text=f"Invalid input: {str(e)}", font=("Arial", 11)).pack(pady=20)
            CTkButton(error_window, text="OK", command=error_window.destroy).pack()

    CTkButton(param_window, text="Save Parameters", command=save_hyperparams, fg_color="green", height=35).pack(pady=15)


#GUI
default_Page = CTkFrame(app, width=300)
default_Page.grid(row=0, column=0, sticky="nsew")
default_Page_menu = CTkFrame(default_Page)
default_Page_menu.rowconfigure(1, weight=800)
default_Page_video = CTkFrame(default_Page, width=800, height=600)
default_Page_menu.grid(row=0, column=0, sticky="nsew")
default_Page_video.grid(row=0, column=1, sticky="e")

lable1 = CTkLabel(default_Page_menu, text="HOME PAGE", width=60, height=10)
lable1.grid(row=0, column=0)
space_lable = CTkLabel(default_Page_menu, text="", width=200)
space_lable.grid(row=1, column=0)
default_Page_lable = CTkLabel(default_Page_video, text="", width=800, height=570)
image = Image.open('framework/3d-face-recognition-icon-png.webp')
photo_image = CTkImage(image, size=(400, 400))
default_Page_lable.photo_image = photo_image
default_Page_lable.configure(image=photo_image)
default_Page_lable.pack()
default_Page_lable1 = CTkLabel(default_Page_video, text="Made With ❤️ by akhil838")
default_Page_lable1.pack(side=BOTTOM)

# Buttons on home page
button1 = CTkButton(default_Page_menu, text="Recognise (Test)", command=lambda: (recog.tkraise()))
button1.grid(row=2, column=0, pady=5)
button6 = CTkButton(default_Page_menu, text="Add Faces (Train)", command=lambda: add_face.tkraise())
button6.grid(row=3, column=0, pady=5)
button7 = CTkButton(default_Page_menu, text="Show Dataset Stats", command=show_dataset_stats)
button7.grid(row=4, column=0, pady=5)

# FACE RECOGNITION FRAME
recog = CTkFrame(app)
recog_menu = CTkFrame(recog)
recog_menu.configure(width=300)
recog_menu.rowconfigure(3, weight=40)
recog_video = CTkFrame(recog, width=800, height=600)

recog.grid(row=0, column=0, sticky="nsew")
recog_menu.grid(row=0, column=0, sticky="nsew")
recog_video.grid(row=0, column=1, sticky="e")

lable2 = CTkLabel(recog_menu, text="Face Recognition", width=60, height=10)
lable2.grid(row=0, column=0, sticky="n")
lable_video = CTkLabel(recog_video, text="", width=800, height=600)
lable_video.pack()
lable0 = CTkLabel(recog_menu, text="Select Camera ", height=10)
lable0.grid(row=1, column=0, pady=5)
cam_box1 = CTkComboBox(recog_menu, state='readonly', values=camera_list)
cam_box1.grid(row=2, column=0)

# Model selection and comparison
model_label = CTkLabel(recog_menu, text="Select Model", height=10)
model_label.grid(row=3, column=0, pady=5)
model_choice = CTkComboBox(recog_menu, state='readonly', values=["knn", "svm", "rf"])
model_choice.grid(row=4, column=0)
model_choice.set("knn")


def on_model_change(*args):
    get_current_model()


model_choice.bind("<<ComboboxSelected>>", on_model_change)

compare_btn = CTkButton(recog_menu, text="Compare Models Accuracy", command=compare_models)
compare_btn.grid(row=5, column=0, pady=5)

# Hyperparameters button
hyperparam_btn = CTkButton(recog_menu, text="Configure Hyperparameters", command=open_hyperparam_window,
                           fg_color="orange")
hyperparam_btn.grid(row=6, column=0, pady=5)

# Add stats button to recognition page
stats_btn = CTkButton(recog_menu, text="Dataset Statistics", command=show_dataset_stats)
stats_btn.grid(row=7, column=0, pady=5)

space_lable = CTkLabel(recog_menu, text="", width=200)
space_lable.grid(row=8, column=0)


# Helper functions for buttons
def disablebutton(button):
    button.configure(state=DISABLED)


def enablebutton(button):
    button.configure(state=NORMAL)
    global img_path
    img_path = ''


def testing():
    global vid
    disablebutton(button4)
    vid = cv2.VideoCapture(camera_list.index(cam_box1.get()), cv2.CAP_DSHOW)
    width, height = 800, 600
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    model = get_current_model()
    if model is None:
        # Show error if no data
        lable_video.configure(text="No training data! Please add faces first.")
        enablebutton(button4)
        return

    def test_video():
        _, frame = vid.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(grey, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = model.predict(resized_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = CTkImage(captured_image, size=(800, 600))
        lable_video.photo_image = photo_image
        lable_video.configure(image=photo_image)
        lable_video.after(5, test_video)

    test_video()


button4 = CTkButton(recog_menu, text="Turn On Camera", command=testing)
button4.grid(row=9, column=0, pady=5)
stop_button_recog = CTkButton(recog_menu, text="Stop", command=lambda: (enablebutton(button4), vid.release()))
stop_button_recog.grid(row=10, column=0, pady=5)

button2 = CTkButton(recog_menu, text="Go back to Home", command=lambda: default_Page.tkraise())
button2.grid(row=11, column=0, pady=5)

# ADD FACES FRAME
add_face = CTkFrame(app)
add_face_menu = CTkFrame(add_face)
add_face_video = CTkFrame(add_face, width=800, height=600)
add_face_menu.rowconfigure(5, weight=40)

add_face.grid(row=0, column=0, sticky="nsew")
add_face_menu.grid(row=0, column=0, sticky="nsew")
add_face_video.grid(row=0, column=1, sticky="e")

lable3 = CTkLabel(add_face_menu, text="Train a New Face", width=60, height=10)
lable3.grid(row=0, column=0, sticky="n")
lable_train = CTkLabel(add_face_video, text="", width=800, height=600)
lable_train.pack()
space_lable1 = CTkLabel(add_face_menu, text="", width=200)
space_lable1.grid(row=5, column=0)

button3 = CTkButton(add_face_menu, text="Go back to Home", command=lambda: default_Page.tkraise())
button3.grid(row=14, column=0, pady=5)
name_lable = CTkLabel(add_face_menu, text="Enter Name")
name_lable.grid(row=10, column=0)
name_input = CTkTextbox(add_face_menu, height=5, width=140)
name_input.grid(row=11, column=0)
lable0 = CTkLabel(add_face_menu, text="Select Camera ", height=10)
lable0.grid(row=2, column=0, pady=5)

cam_box2 = CTkComboBox(add_face_menu, state='readonly', values=camera_list)
cam_box2.grid(row=3, column=0)
var1 = IntVar()


def train():
    global vid
    disablebutton(button5)
    global img_path
    global faces_data
    global i
    global name
    i = 0
    faces_data = []
    name = name_input.get("1.0", "end-1c")
    if not name.strip():
        enablebutton(button5)
        return
    img_path = f'./data/Train_faces/{name}'

    def train_video():
        global i
        global faces_data
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i = i + 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = CTkImage(captured_image, size=(800, 600))
        lable_train.photo_image = photo_image
        lable_train.configure(image=photo_image)

        if len(faces_data) < 100:
            lable_train.after(5, train_video)
        else:
            faces_data = np.asarray(faces_data)
            faces_data = faces_data.reshape(100, -1)

            if 'names.pkl' not in os.listdir('data/'):
                names = [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)
            else:
                with open('data/names.pkl', 'rb') as f:
                    names = pickle.load(f)
                names = names + [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)

            if 'faces_data.pkl' not in os.listdir('data/'):
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)
            else:
                with open('data/faces_data.pkl', 'rb') as f:
                    faces = pickle.load(f)
                faces = np.append(faces, faces_data, axis=0)
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces, f)
            enablebutton(button5)
            vid.release()
            # Reset current model so that it gets retrained when needed
            global current_model
            current_model = None
            # Show success message
            success_window = CTkToplevel(app)
            success_window.title("Success")
            success_window.geometry("300x100")
            CTkLabel(success_window, text=f"Training complete!\nAdded 100 samples for {name}").pack(pady=20)
            CTkButton(success_window, text="OK", command=success_window.destroy).pack()

    def train_image():
        global i
        global j
        global faces_data

        frame = cv2.imread(img_path + f'/{j}.jpg')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100:
                faces_data.append(resized_img)
            i = i + 1
            if len(faces_data) % (100 // len(os.listdir(img_path))) == 0:
                j += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = CTkImage(captured_image, size=(800, 600))
        lable_train.photo_image = photo_image
        lable_train.configure(image=photo_image)

        if len(faces_data) < 100:
            lable_train.after(5, train_image)
        else:
            faces_data = np.asarray(faces_data)
            faces_data = faces_data.reshape(100, -1)

            if 'names.pkl' not in os.listdir('data/'):
                names = [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)
            else:
                with open('data/names.pkl', 'rb') as f:
                    names = pickle.load(f)
                names = names + [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)

            if 'faces_data.pkl' not in os.listdir('data/'):
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)
            else:
                with open('data/faces_data.pkl', 'rb') as f:
                    faces = pickle.load(f)
                faces = np.append(faces, faces_data, axis=0)
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces, f)
            enablebutton(button5)
            current_model = None
            success_window = CTkToplevel(app)
            success_window.title("Success")
            success_window.geometry("300x100")
            CTkLabel(success_window, text=f"Training complete!\nAdded 100 samples for {name}").pack(pady=20)
            CTkButton(success_window, text="OK", command=success_window.destroy).pack()

    if var1.get() == 0:
        vid = cv2.VideoCapture(camera_list.index(cam_box2.get()), cv2.CAP_DSHOW)
        width, height = 800, 600
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        train_video()
    else:
        global j
        j = 1
        if not os.path.exists(img_path):
            enablebutton(button5)
            return
        train_image()


button5 = CTkButton(add_face_menu, text="Train", command=train)
button5.grid(row=12, column=0, pady=5)
button10 = CTkCheckBox(add_face_menu, variable=var1, onvalue=1, offvalue=0, text="Train From Pics")
button10.grid(row=4, column=0, pady=5)
stop_button_train = CTkButton(add_face_menu, text="Stop", command=lambda: (enablebutton(button5), vid.release()))
stop_button_train.grid(row=13, column=0, pady=5)

default_Page.tkraise()
app.title("Facial Recognition and Emotion Predictor")
app.resizable(True, False)
app.mainloop()