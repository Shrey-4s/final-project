let model = null;
let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let startBtn = document.getElementById('startBtn');
let uploadBtn = document.getElementById('uploadBtn');
let imageInput = document.getElementById('imageInput');
let isDetecting = false;
let isImageInput = false;

// Load model
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Model loaded!");
}
loadModel();

// Setup webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Webcam error:", err);
    });

// Prediction function for both image and video
async function predict(inputImage) {
    if (!model) return;

    let img = inputImage;
    if (video && !isImageInput) {
        // Capture frame if video is being used
        ctx.drawImage(video, 0, 0, 640, 480);
        img = ctx.getImageData(0, 0, 640, 480);
    }

    // Preprocess image (resize to match model input size)
    let tensor = tf.browser.fromPixels(img)
        .resizeBilinear([150, 150]) // Adjusted resize to 150x150, as expected by the model
        .toFloat()
        .div(255.0) // Normalize pixel values
        .expandDims(); // Add batch dimension (1, 150, 150, 3)

    // Log tensor shape for debugging
    console.log("Tensor shape:", tensor.shape);

    // Predict the result
    let prediction = await model.predict(tensor).data();
    let result = prediction[0] > 0.5 ? 'Mask' : 'No Mask'; // Adjust threshold for classification

    // Display prediction result
    document.getElementById('prediction').innerText = `Prediction: ${result}`;
}

// Toggle detection for webcam feed
startBtn.addEventListener('click', () => {
    isDetecting = !isDetecting;
    startBtn.innerText = isDetecting ? 'Stop' : 'Start';
    if (isDetecting) {
        isImageInput = false;
        predict();
    }
});

// Image upload functionality
uploadBtn.addEventListener('click', () => {
    imageInput.click();
});

// When an image is selected
imageInput.addEventListener('change', (event) => {
    let file = event.target.files[0];
    if (file) {
        let reader = new FileReader();
        reader.onload = function (e) {
            let imgElement = new Image();
            imgElement.src = e.target.result;
            imgElement.onload = function () {
                isImageInput = true;
                document.getElementById('prediction').innerText = "Prediction: Processing...";
                predict(imgElement);
            };
        };
        reader.readAsDataURL(file);
    }
});
