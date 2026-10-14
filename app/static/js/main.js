/**
 * MEDIFLORA AI | Medicinal Flower Recognition & Explainable AI
 * Main Client Application Script
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const changeFileBtn = document.getElementById('changeFileBtn');
    const predictBtn = document.getElementById('predictBtn');
    
    const alertBox = document.getElementById('alertBox');
    const alertMessage = document.getElementById('alertMessage');
    
    const loadingState = document.getElementById('loadingState');
    const loadingStatusText = document.getElementById('loadingStatusText');
    const resultsState = document.getElementById('resultsState');
    
    const recognizedContent = document.getElementById('recognizedContent');
    const unrecognizedContent = document.getElementById('unrecognizedContent');
    
    const predictedFlower = document.getElementById('predictedFlower');
    const scientificName = document.getElementById('scientificName');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const botanicalDescription = document.getElementById('botanicalDescription');
    const traditionalUses = document.getElementById('traditionalUses');
    const gradcamImage = document.getElementById('gradcamImage');
    
    const unrecognizedMessage = document.getElementById('unrecognizedMessage');
    const supportedClassesGrid = document.getElementById('supportedClassesGrid');
    const topPredictionsList = document.getElementById('topPredictionsList');
    const unrecognizedCandidatesList = document.getElementById('unrecognizedCandidatesList');
    
    let selectedFile = null;
    let loadingInterval = null;

    // Master List of 31 Supported Species
    const SUPPORTED_CLASSES = [
        "Adathoda", "Allium_Cepa", "Banana", "Black_cumin", "Bush Clock Vine",
        "Champaka", "Chitrak", "Common Lanthana", "Crown Flower", "Datura",
        "Four O' Clock Flower", "Hibiscus", "Honey Suckle", "Indian Mallow",
        "Jatropha", "Malabar Melastome", "Marigold", "Nagapoovu", "Nityakalyani",
        "Pinwheel", "Rose", "Shankupushpam", "Spider Lily", "Sunflower",
        "Sweet_pea", "Thechi", "Thumba", "Touch Me Not", "Tridax procumbens",
        "Wild potato vine", "Yellow Daisy"
    ];

    // Populate Catalog Modal Grid
    if (supportedClassesGrid) {
        supportedClassesGrid.innerHTML = SUPPORTED_CLASSES.map((c, i) => `
            <div class="col">
                <div class="p-2 border rounded-3 bg-white d-flex align-items-center gap-2 shadow-xs">
                    <span class="badge bg-emerald-subtle text-emerald-dark border">${i + 1}</span>
                    <span class="fw-semibold text-slate-800">${c}</span>
                </div>
            </div>
        `).join('');
    }

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File Input Click
    dropZone.addEventListener('click', (e) => {
        if (e.target.id === 'changeFileBtn' || e.target.closest('#changeFileBtn')) {
            fileInput.click();
        } else if (previewContainer.classList.contains('d-none')) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Handle File Selection & Preview
    function handleFileSelect(file) {
        hideAlert();
        
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type.toLowerCase())) {
            showAlert('Unsupported file format. Please upload a valid JPG, JPEG, or PNG flower image.');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('d-none');
            previewContainer.classList.remove('d-none');
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (changeFileBtn) {
        changeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    // Predict Button Handler
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            showAlert('Please select a flower image first.');
            return;
        }

        hideAlert();
        startLoadingAnimation();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An error occurred during flower prediction.');
            }

            stopLoadingAnimation();
            renderResults(data);

            // Smooth scroll to results
            resultsState.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error('Prediction Error:', error);
            stopLoadingAnimation();
            showAlert(error.message || 'Failed to connect to prediction service.');
        }
    });

    // Animated Loading Messages
    function startLoadingAnimation() {
        resultsState.classList.add('d-none');
        loadingState.classList.remove('d-none');

        const steps = [
            "Preprocessing input image (224x224 RGB)...",
            "Running MobileNetV2 deep neural inference...",
            "Matching against 31 supported medicinal classes...",
            "Computing Grad-CAM visual feature heatmaps..."
        ];
        let stepIdx = 0;
        loadingStatusText.textContent = steps[0];

        loadingInterval = setInterval(() => {
            stepIdx = (stepIdx + 1) % steps.length;
            loadingStatusText.textContent = steps[stepIdx];
        }, 750);
    }

    function stopLoadingAnimation() {
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
        loadingState.classList.add('d-none');
    }

    // Render Results
    function renderResults(data) {
        loadingState.classList.add('d-none');
        resultsState.classList.remove('d-none');

        if (data.recognized) {
            // Recognized Flower View
            predictedFlower.textContent = data.predicted_flower;
            scientificName.textContent = data.scientific_name !== 'N/A' ? `Scientific Name: ${data.scientific_name}` : `Common Name: ${data.common_name}`;
            confidenceBadge.textContent = `${data.confidence.toFixed(2)}%`;

            botanicalDescription.textContent = data.short_botanical_description || 'Botanical description available in knowledge database.';
            traditionalUses.textContent = data.traditional_uses || 'Traditional reported uses documented in database.';

            // Render Top Prediction Bars
            renderPredictionBars(data.top_predictions, topPredictionsList);

            // Render Grad-CAM Image
            gradcamImage.src = data.gradcam_image_url;

            recognizedContent.classList.remove('d-none');
            unrecognizedContent.classList.add('d-none');

        } else {
            // Unrecognized / Out-of-Distribution Input View
            unrecognizedMessage.textContent = data.message || 'This image does not sufficiently match any of the 31 supported flower classes.';

            // Render candidate predictions that were below threshold
            renderPredictionBars(data.top_predictions, unrecognizedCandidatesList);

            recognizedContent.classList.add('d-none');
            unrecognizedContent.classList.remove('d-none');
        }
    }

    function renderPredictionBars(predictions, container) {
        if (!container || !predictions || predictions.length === 0) return;
        container.innerHTML = predictions.slice(0, 5).map((p, idx) => `
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="fw-semibold text-slate-800 small">${idx + 1}. ${p.class_name}</span>
                    <span class="fw-bold small ${idx === 0 ? 'text-emerald-dark' : 'text-slate-500'}">${p.probability.toFixed(2)}%</span>
                </div>
                <div class="progress rounded-pill" style="height: 7px; background-color: #f1f5f9;">
                    <div class="progress-bar rounded-pill ${idx === 0 ? 'bg-emerald-main' : 'bg-slate-400'}" 
                         role="progressbar" 
                         style="width: ${Math.max(p.probability, 1.5)}%" 
                         aria-valuenow="${p.probability}" 
                         aria-valuemin="0" 
                         aria-valuemax="100"></div>
                </div>
            </div>
        `).join('');
    }

    function showAlert(msg) {
        alertMessage.textContent = msg;
        alertBox.classList.remove('d-none');
    }

    function hideAlert() {
        alertBox.classList.add('d-none');
    }
});
