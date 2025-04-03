document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imageNumber = document.getElementById('imageNumber');
    const previewImage = document.getElementById('previewImage');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');
    const retrainButton = document.getElementById('retrainButton');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const retrainZipUpload = document.getElementById('retrainZipUpload');
    const uploadRetrainDataButton = document.getElementById('uploadRetrainDataButton');

    let selectedImage = null;

    imagePlaceholder.style.display = "flex";

    imageUpload.addEventListener('change', (event) => {
        selectedImage = event.target.files[0];
        imageNumber.value = "";
        if (selectedImage) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(selectedImage);
        } else {
            previewImage.src = "";
        }
        imagePlaceholder.style.display = "none";
        previewImage.style.display = "block";
    });

    imageNumber.addEventListener('change', () => {
        selectedImage = imageNumber.value;
        imageUpload.value = "";
        if (selectedImage) {
            fetch(`http://127.0.0.1:5000/image/${selectedImage}`) // Updated fetch
                .then(response => response.blob())
                .then(blob => {
                    previewImage.src = URL.createObjectURL(blob);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Image not found.');
                    previewImage.src = "";
                });
        } else {
            previewImage.src = "";
        }
        imagePlaceholder.style.display = "none";
        previewImage.style.display = "block";
    });

    predictButton.addEventListener('click', () => {
        predictionResult.textContent = ''; // Clear previous result
        console.log("Predict button clicked");

        if (imageUpload.files.length > 0) {
            console.log("Image upload detected");
            const formData = new FormData();
            formData.append('image', imageUpload.files[0]);

            fetch('http://127.0.0.1:5000/predict_upload', { 
                method: 'POST',
                body: formData,
            })
                .then(response => {
                    console.log("API Response received", response);
                    return response.json();
                })
                .then(data => {
                    console.log("Parsed JSON:", data);
                    if (data.prediction) {
                        predictionResult.textContent = `Prediction: ${data.prediction}`;
                    } else if (data.error) {
                        predictionResult.textContent = `Prediction failed: ${data.error}`;
                    } else {
                        predictionResult.textContent = 'Prediction failed.';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    predictionResult.textContent = 'Prediction failed.';
                });
        } else if (imageNumber.value) {
            fetch('http://127.0.0.1:5000/predict_lib', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image_id: imageNumber.value }),
            })
                .then(response => response.json())
                .then(data => {
                    if (data.prediction) {
                        predictionResult.textContent = `Prediction: ${data.prediction}`;
                    } else if (data.error) {
                        predictionResult.textContent = `Prediction failed: ${data.error}`;
                    } else {
                        predictionResult.textContent = 'Prediction failed.';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    predictionResult.textContent = 'Prediction failed.';
                });
        } else {
            alert("Please select an image or enter an image number.");
        }
    });

    retrainZipUpload.addEventListener('change', () => {
        if (retrainZipUpload.files.length > 0) {
            const file = retrainZipUpload.files[0];
            const formData = new FormData();
            formData.append('zip_file', file);

            fetch('http://127.0.0.1:5000/upload_retrain_data', {
                method: 'POST',
                body: formData,
            })
                .then(response => response.json())
                .then(data => {
                    if (data.message) {
                        document.getElementById('uploadMessage').textContent = data.message + ". Click the retrain button to trigger retraining process."; //display message.
                    } else {
                        document.getElementById('uploadMessage').textContent = 'Upload failed.'; //display message.
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('uploadMessage').textContent = 'Upload failed.'; //display message.
                });
        } else {
            document.getElementById('uploadMessage').textContent = ""; // Clear message if no file.
        }
    });

    uploadRetrainDataButton.addEventListener('click', () => {
        retrainZipUpload.click();
    });

    retrainButton.addEventListener('click', () => {
        fetch('http://127.0.0.1:5000/retrain', {
            method: 'POST',
        })
            .then(response => response.json())
            .then(data => {
                if (data.retrain_id) {
                    monitorRetraining(data.retrain_id);
                } else {
                    alert(data.message || 'Retrain failed.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Retrain failed.');
            });
    });
    
    function monitorRetraining(retrainId) {
        const intervalId = setInterval(() => {
            fetch(`http://127.0.0.1:5000/retrain_status/${retrainId}`, { // Updated fetch
                method: 'GET',
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(intervalId);
                        alert(data.message);
                        if (data.metrics && data.metrics.metrics_table) {
                            displayRetrainMetricsTable(data.metrics.metrics_table);
                        }
                    }
                    console.log(`Retraining status: ${data.status}, progress: ${data.progress}%`);
                })
                .catch(error => {
                    console.error('Error:', error);
                    clearInterval(intervalId);
                    alert('Monitoring failed.');
                });
        }, 5000);
    }

    function displayRetrainMetricsTable(tableString) {
        let metricsContainer = document.getElementById('retrainMetrics');
        if (!metricsContainer) {
            metricsContainer = document.createElement('div');
            metricsContainer.id = 'retrainMetrics';
            document.querySelector('.retrain-section').appendChild(metricsContainer);
        } else {
            metricsContainer.innerHTML = "";
        }
        const pre = document.createElement('pre');
        pre.textContent = tableString;
        metricsContainer.appendChild(pre);
    }

    previewImage.addEventListener("load", () => {
        imagePlaceholder.style.display = "none";
        previewImage.style.display = "block";
    });
});