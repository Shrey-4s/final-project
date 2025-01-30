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

    // Capture frame
    ctx.drawImage(video, 0, 0, 640, 480);
    let img = ctx.getImageData(0, 0, 640, 480);

    // Preprocess image (adjust dimensions to match model input)
    let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224]) // Example size, adjust as needed
        .toFloat()
        .div(255.0)
        .expandDims();

    // Predict
    let prediction = await model.predict(tensor).data();
    let result = prediction[0] > 0.5 ? 'Mask' : 'No Mask'; // Adjust threshold

    // Display result
    document.getElementById('prediction').innerText = `Prediction: ${result}`;

    if (isDetecting) requestAnimationFrame(predict);
}

// Toggle detection
startBtn.addEventListener('click', () => {
    isDetecting = !isDetecting;
    startBtn.innerText = isDetecting ? 'Stop' : 'Start';
    if (isDetecting) predict();
});