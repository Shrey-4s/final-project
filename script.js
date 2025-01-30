let model = null;
let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let startBtn = document.getElementById('startBtn');
let isDetecting = false;

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
    });

// Prediction function
async function predict() {
    if (!model) return;

    // Capture frame from webcam
    ctx.drawImage(video, 0, 0, 640, 480);
    let img = ctx.getImageData(0, 0, 640, 480);

    // Preprocess image (resize to match model input size)
    let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([150, 150]) // Adjusted to 150x150, change based on model requirements
        .toFloat()
        .div(255.0) // Normalize pixel values
        .expandDims(); // Add batch dimension

    // Predict the result
    let prediction = await model.predict(tensor).data();
    let result = prediction[0] > 0.5 ? 'Mask' : 'No Mask'; // Adjust threshold for classification

    // Display prediction result
    document.getElementById('prediction').innerText = `Prediction: ${result}`;

    if (isDetecting) requestAnimationFrame(predict); // Continue detecting if flag is true
}


// Toggle detection
startBtn.addEventListener('click', () => {
    isDetecting = !isDetecting;
    startBtn.innerText = isDetecting ? 'Stop' : 'Start';
    if (isDetecting) predict();
});