document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imageNumber = document.getElementById('imageNumber');
    const previewImage = document.getElementById('previewImage');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');
    const retrainButton = document.getElementById('retrainButton');
    const imagePlaceholder = document.getElementById('imagePlaceholder');

    let selectedImage = null;

    imageUpload.addEventListener('change', (event) => {
        selectedImage = event.target.files[0];
        imageNumber.value = ""; // Clear library selection
        if (selectedImage) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(selectedImage);
        }
    });

    imageNumber.addEventListener('change', () => {
        selectedImage = imageNumber.value;
        imageUpload.value = ""; // Clear file upload
        if (selectedImage) {
            fetch(`/image/${selectedImage}`)
                .then(response => response.blob())
                .then(blob => {
                    previewImage.src = URL.createObjectURL(blob);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Image not found.');
                    previewImage.src = "";
                });
        }
    });

    predictButton.addEventListener('click', () => {
        if (imageUpload.files.length > 0) {
            // Image upload
            const formData = new FormData();
            formData.append('image', imageUpload.files[0]);
    
            fetch('/predict_upload', {
                method: 'POST',
                body: formData,
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
        } else if (imageNumber.value) {
            // Library selection
            fetch('/predict_lib', {
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

    uploadRetrainDataButton.addEventListener('click', () => {
        const fileInput = document.getElementById('retrainDataUpload');
        if (!fileInput.files[0]) {
            alert("Please select a ZIP file to upload.");
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('zip_file', file);

        fetch('/upload_retrain_images', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.retrain_id) {
                monitorRetraining(data.retrain_id);
            } else {
                alert(data.message || 'Upload failed.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Upload failed.');
        });
    });

    retrainButton.addEventListener('click', () => {
        fetch('/retrain', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.retrain_id) {
                monitorRetraining(data.retrain_id);
            } else {
                alert(data.message || 'Retraining failed.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Retraining failed.');
        });
    });

    function monitorRetraining(retrainId) {
        const intervalId = setInterval(() => {
            fetch(`/retrain_status/${retrainId}`)
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
        }, 5000); // Poll every 5 seconds
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

    imageNumber.addEventListener("change", () => {
        imagePlaceholder.style.display = "none";
    });

    imageUpload.addEventListener("change", () => {
        imagePlaceholder.style.display = "none";
    });

    if (previewImage.src === "") {
        imagePlaceholder.style.display = "flex";
        previewImage.style.display = "none";
    }
});