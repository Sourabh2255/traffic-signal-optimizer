/**
 * AI Adaptive Traffic Signal Control - Frontend Controller
 * 
 * Manages:
 * 1. 4-way traffic camera image uploads (North, East, South, West)
 * 2. Optional emergency siren WAV upload with direction selector
 * 3. Client-side file validation & previews
 * 4. Asynchronous traffic analysis submission & progress indication
 * 5. Dynamic results dashboard rendering with color-coded status badges
 * 6. Interactive 4-way signal simulation integration
 * 7. 1-click College Demo presets
 */

(function () {
    "use strict";

    // Application state
    const ROADS = ["north", "east", "south", "west"];
    const ROAD_NAMES = {
        north: "North",
        east: "East",
        south: "South",
        west: "West"
    };
    const ROAD_ICONS = {
        north: "⬆",
        east: "➡",
        south: "⬇",
        west: "⬅"
    };

    const filesState = {
        north: null,
        east: null,
        south: null,
        west: null,
        siren: null,
    };

    // DOM Elements
    let analyzeBtn;
    let validationMsg;
    let uploadSummaryBadge;
    let loadingOverlay;
    let loadingStepText;
    let resultsSection;
    let simulationSection;
    let resultsCardsContainer;
    let emergencyAlertBanner;
    let bannerHeading;
    let bannerReason;
    let bannerBadge;
    let bannerIcon;
    let cycleBarsContainer;
    let totalCycleBadge;
    let lightboxModal;
    let lightboxImg;
    let lightboxTitle;

    // Emergency Analysis Panel Elements (Task 5)
    let emergencyAnalysisPanel;
    let emPanelStatusBadge;
    let emAuditDirection;
    let emAuditAmbulance;
    let emAuditFiretruck;
    let emAuditConfidence;
    let emAuditSiren;
    let emAuditSirenConf;
    let emAuditConfirmation;
    let emAuditReason;
    let emPreviewImg;
    let emPreviewEmpty;
    let btnInspectEmAnnotated;

    // Helper: format bytes
    function formatBytes(bytes) {
        if (!bytes || bytes === 0) return "0 KB";
        const k = 1024;
        const dm = 1;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
    }

    // Initialize DOM handlers
    document.addEventListener("DOMContentLoaded", () => {
        analyzeBtn = document.getElementById("analyze-btn");
        validationMsg = document.getElementById("validation-msg");
        uploadSummaryBadge = document.getElementById("upload-summary-badge");
        loadingOverlay = document.getElementById("loading-overlay");
        loadingStepText = document.getElementById("loading-step-text");
        resultsSection = document.getElementById("results-section");
        simulationSection = document.getElementById("simulation-section");
        resultsCardsContainer = document.getElementById("results-cards-container");
        emergencyAlertBanner = document.getElementById("emergency-alert-banner");
        bannerHeading = document.getElementById("banner-heading");
        bannerReason = document.getElementById("banner-reason");
        bannerBadge = document.getElementById("banner-badge");
        bannerIcon = document.getElementById("banner-icon");
        cycleBarsContainer = document.getElementById("cycle-bars-container");
        totalCycleBadge = document.getElementById("total-cycle-badge");
        lightboxModal = document.getElementById("lightbox-modal");
        lightboxImg = document.getElementById("lightbox-img");
        lightboxTitle = document.getElementById("lightbox-title");

        // Emergency Analysis Panel Elements (Task 5)
        emergencyAnalysisPanel = document.getElementById("emergency-analysis-panel");
        emPanelStatusBadge = document.getElementById("em-panel-status-badge");
        emAuditDirection = document.getElementById("em-audit-direction");
        emAuditAmbulance = document.getElementById("em-audit-ambulance");
        emAuditFiretruck = document.getElementById("em-audit-firetruck");
        emAuditConfidence = document.getElementById("em-audit-confidence");
        emAuditSiren = document.getElementById("em-audit-siren");
        emAuditSirenConf = document.getElementById("em-audit-siren-conf");
        emAuditConfirmation = document.getElementById("em-audit-confirmation");
        emAuditReason = document.getElementById("em-audit-reason");
        emPreviewImg = document.getElementById("em-preview-img");
        emPreviewEmpty = document.getElementById("em-preview-empty");
        btnInspectEmAnnotated = document.getElementById("btn-inspect-em-annotated");

        if (btnInspectEmAnnotated && emPreviewImg) {
            btnInspectEmAnnotated.addEventListener("click", () => {
                if (emPreviewImg.src && emPreviewImg.src !== "" && !emPreviewImg.src.endsWith("/")) {
                    openLightbox(emPreviewImg.src, "Emergency Vehicle Detection Preview");
                }
            });
        }
        if (emPreviewImg) {
            emPreviewImg.addEventListener("click", () => {
                if (emPreviewImg.src && emPreviewImg.src !== "" && !emPreviewImg.src.endsWith("/")) {
                    openLightbox(emPreviewImg.src, "Emergency Vehicle Detection Preview");
                }
            });
        }

        // Set up 4 directional image inputs
        ROADS.forEach(road => {
            setupImageUpload(road);
        });

        // Set up optional audio upload
        setupAudioUpload();

        // Set up analyze button click
        if (analyzeBtn) {
            analyzeBtn.addEventListener("click", handleAnalyzeTraffic);
        }

        // Set up preset demo buttons
        const btnDemoEmergency = document.getElementById("btn-demo-emergency");
        const btnDemoNormal = document.getElementById("btn-demo-normal");
        const btnClearAll = document.getElementById("btn-clear-all");

        if (btnDemoEmergency) {
            btnDemoEmergency.addEventListener("click", () => loadDemoPreset("emergency"));
        }
        if (btnDemoNormal) {
            btnDemoNormal.addEventListener("click", () => loadDemoPreset("normal"));
        }
        if (btnClearAll) {
            btnClearAll.addEventListener("click", clearAllInputs);
        }

        // Lightbox modal close handlers
        const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
        if (lightboxCloseBtn) {
            lightboxCloseBtn.addEventListener("click", closeLightbox);
        }
        if (lightboxModal) {
            lightboxModal.addEventListener("click", (e) => {
                if (e.target === lightboxModal) closeLightbox();
            });
        }
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && lightboxModal && lightboxModal.style.display !== "none") {
                closeLightbox();
            }
        });

        updateValidationStatus();
    });

    // --------------------------------------------------------------------------
    // Image Upload Handling (North, East, South, West)
    // --------------------------------------------------------------------------
    function setupImageUpload(road) {
        const dropzone = document.getElementById(`dropzone-${road}`);
        const input = document.getElementById(`file-${road}`);

        if (!dropzone || !input) return;

        // Click on dropzone triggers hidden file input
        dropzone.addEventListener("click", () => input.click());

        // File input changed
        input.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                processSelectedImage(road, e.target.files[0]);
            }
        });

        // Drag & drop events
        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files[0]) {
                processSelectedImage(road, dt.files[0]);
            }
        });
    }

    function processSelectedImage(road, file) {
        // Validate extension
        const validExtensions = [".jpg", ".jpeg", ".png"];
        const lowerName = file.name.toLowerCase();
        const hasValidExt = validExtensions.some(ext => lowerName.endsWith(ext));

        if (!hasValidExt) {
            alert(`Invalid file format for ${ROAD_NAMES[road]} direction. Please upload a JPG, JPEG, or PNG image.`);
            return;
        }

        filesState[road] = file;

        // Update preview UI
        const dropzone = document.getElementById(`dropzone-${road}`);
        const preview = document.getElementById(`preview-${road}`);
        const img = document.getElementById(`img-${road}`);
        const nameEl = document.getElementById(`name-${road}`);
        const sizeEl = document.getElementById(`size-${road}`);
        const statusTag = document.getElementById(`status-tag-${road}`);

        if (img) {
            const objectUrl = URL.createObjectURL(file);
            img.src = objectUrl;
        }
        if (nameEl) nameEl.textContent = file.name;
        if (sizeEl) sizeEl.textContent = formatBytes(file.size);

        if (dropzone) dropzone.style.display = "none";
        if (preview) preview.style.display = "flex";
        if (statusTag) {
            statusTag.textContent = "Ready ✓";
            statusTag.className = "file-status-tag tag-ready";
        }

        updateValidationStatus();
    }

    // Expose triggerFileInput & removeFile to window
    window.triggerFileInput = function (road) {
        const input = document.getElementById(`file-${road}`);
        if (input) input.click();
    };

    window.removeFile = function (road) {
        filesState[road] = null;
        const input = document.getElementById(`file-${road}`);
        if (input) input.value = "";

        const dropzone = document.getElementById(`dropzone-${road}`);
        const preview = document.getElementById(`preview-${road}`);
        const img = document.getElementById(`img-${road}`);
        const statusTag = document.getElementById(`status-tag-${road}`);

        if (img) img.src = "";
        if (preview) preview.style.display = "none";
        if (dropzone) dropzone.style.display = "flex";
        if (statusTag) {
            statusTag.textContent = "Required";
            statusTag.className = "file-status-tag";
        }

        updateValidationStatus();
    };

    // --------------------------------------------------------------------------
    // Siren Audio Upload Handling
    // --------------------------------------------------------------------------
    function setupAudioUpload() {
        const dropzone = document.getElementById("dropzone-audio");
        const input = document.getElementById("file-siren");

        if (!dropzone || !input) return;

        dropzone.addEventListener("click", () => input.click());

        input.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                processSelectedAudio(e.target.files[0]);
            }
        });

        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files[0]) {
                processSelectedAudio(dt.files[0]);
            }
        });
    }

    function processSelectedAudio(file) {
        if (!file.name.toLowerCase().endsWith(".wav")) {
            alert("Invalid audio format. Emergency siren audio must be a .WAV file.");
            return;
        }

        filesState.siren = file;

        const dropzone = document.getElementById("dropzone-audio");
        const preview = document.getElementById("preview-audio");
        const player = document.getElementById("audio-player");
        const nameEl = document.getElementById("name-audio");
        const sizeEl = document.getElementById("size-audio");

        if (player) {
            player.src = URL.createObjectURL(file);
        }
        if (nameEl) nameEl.textContent = file.name;
        if (sizeEl) sizeEl.textContent = formatBytes(file.size);

        if (dropzone) dropzone.style.display = "none";
        if (preview) preview.style.display = "flex";
    }

    window.removeAudioFile = function () {
        filesState.siren = null;
        const input = document.getElementById("file-siren");
        if (input) input.value = "";

        const dropzone = document.getElementById("dropzone-audio");
        const preview = document.getElementById("preview-audio");
        const player = document.getElementById("audio-player");

        if (player) {
            player.pause();
            player.src = "";
        }
        if (preview) preview.style.display = "none";
        if (dropzone) dropzone.style.display = "flex";
    };

    // --------------------------------------------------------------------------
    // Validation Logic
    // --------------------------------------------------------------------------
    function updateValidationStatus() {
        const uploadedCount = ROADS.filter(r => filesState[r] !== null).length;
        const allUploaded = uploadedCount === 4;

        if (uploadSummaryBadge) {
            uploadSummaryBadge.textContent = `${uploadedCount} / 4 Directions Ready`;
            if (allUploaded) {
                uploadSummaryBadge.className = "badge-status badge-green";
            } else {
                uploadSummaryBadge.className = "badge-status badge-amber";
            }
        }

        if (validationMsg) {
            if (allUploaded) {
                validationMsg.className = "validation-status-msg valid";
                validationMsg.innerHTML = "✓ All 4 directional approach images are loaded and ready for analysis.";
            } else {
                const missing = ROADS.filter(r => filesState[r] === null).map(r => ROAD_NAMES[r]);
                validationMsg.className = "validation-status-msg";
                validationMsg.innerHTML = `⚠️ Missing images: <strong>${missing.join(", ")}</strong>. All 4 directions must be uploaded before analysis.`;
            }
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = !allUploaded;
        }
    }

    // --------------------------------------------------------------------------
    // 1-Click College Demo Presets
    // --------------------------------------------------------------------------
    async function loadDemoPreset(presetType) {
        try {
            if (loadingOverlay) {
                loadingOverlay.style.display = "flex";
                if (loadingStepText) loadingStepText.textContent = "Fetching demo sample assets from server...";
            }

            const res = await fetch("/api/demo_samples");
            const data = await res.json();

            if (!data.success) {
                throw new Error("Failed to load demo metadata from server");
            }

            const preset = presetType === "emergency" ? data.emergency_demo : data.normal_demo;

            // Load images for all 4 directions
            for (const road of ROADS) {
                const mediaUrl = preset[road];
                const fileRes = await fetch(mediaUrl);
                const blob = await fileRes.blob();
                const filename = mediaUrl.split("/").pop();
                const file = new File([blob], filename, { type: "image/jpeg" });
                processSelectedImage(road, file);
            }

            // Audio handling
            if (preset.siren_audio) {
                const audioRes = await fetch(preset.siren_audio);
                const audioBlob = await audioRes.blob();
                const audioFilename = preset.siren_audio.split("/").pop();
                const audioFile = new File([audioBlob], audioFilename, { type: "audio/wav" });
                processSelectedAudio(audioFile);

                const dirSelect = document.getElementById("siren-direction-select");
                if (dirSelect && preset.siren_direction) {
                    dirSelect.value = preset.siren_direction;
                }
            } else {
                window.removeAudioFile();
            }

            if (loadingOverlay) loadingOverlay.style.display = "none";
        } catch (err) {
            console.error("Demo load error:", err);
            if (loadingOverlay) loadingOverlay.style.display = "none";
            alert("Error loading demo preset: " + err.message);
        }
    }

    function clearAllInputs() {
        ROADS.forEach(r => window.removeFile(r));
        window.removeAudioFile();

        if (resultsSection) resultsSection.style.display = "none";
        if (simulationSection) simulationSection.style.display = "none";
        if (emergencyAnalysisPanel) emergencyAnalysisPanel.style.display = "none";

        if (window.TrafficSimulator && window.TrafficSimulator.stop) {
            window.TrafficSimulator.stop();
        }
    }

    // --------------------------------------------------------------------------
    // Analyze Traffic API Submission
    // --------------------------------------------------------------------------
    async function handleAnalyzeTraffic() {
        // Double check validation
        const missing = ROADS.filter(r => !filesState[r]);
        if (missing.length > 0) {
            alert("Please upload images for all 4 directions: " + missing.map(r => ROAD_NAMES[r]).join(", "));
            return;
        }

        const formData = new FormData();
        ROADS.forEach(road => {
            formData.append(road, filesState[road]);
        });

        if (filesState.siren) {
            formData.append("siren_audio", filesState.siren);
        }

        const sirenDirSelect = document.getElementById("siren-direction-select");
        if (sirenDirSelect) {
            formData.append("siren_direction", sirenDirSelect.value);
        }

        // Show loading overlay with animated progress steps
        if (loadingOverlay) {
            loadingOverlay.style.display = "flex";
            animateLoadingSteps();
        }

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || "Traffic analysis failed on backend");
            }

            renderAnalysisResults(result.data);
        } catch (err) {
            console.error("Analysis execution error:", err);
            alert("Traffic Analysis Failed: " + err.message);
        } finally {
            if (loadingOverlay) {
                loadingOverlay.style.display = "none";
            }
        }
    }

    function animateLoadingSteps() {
        const steps = [
            { id: "step-coco", text: "Processing directional camera feeds with YOLOv8..." },
            { id: "step-emerg", text: "Scanning approaches for ambulances and fire trucks..." },
            { id: "step-siren", text: "Analyzing acoustic siren audio spectrum & classifier..." },
            { id: "step-timing", text: "Computing density demand & optimizing green times..." },
        ];

        let index = 0;
        const interval = setInterval(() => {
            index++;
            if (index < steps.length) {
                if (loadingStepText) loadingStepText.textContent = steps[index].text;
                steps.forEach((s, i) => {
                    const el = document.getElementById(s.id);
                    if (el) {
                        if (i < index) {
                            el.className = "progress-step completed";
                        } else if (i === index) {
                            el.className = "progress-step active";
                        } else {
                            el.className = "progress-step";
                        }
                    }
                });
            } else {
                clearInterval(interval);
            }
        }, 800);
    }

    // --------------------------------------------------------------------------
    // Results Dashboard Rendering
    // --------------------------------------------------------------------------
    function renderAnalysisResults(data) {
        if (!data || !data.results) return;

        // 1. Render Emergency Status Banner
        const sirenAnalysis = data.siren_analysis || { is_siren: false };
        const hasEmergency = Boolean(data.emergency_active && data.emergency_road);
        const emAnalysis = data.emergency_analysis || {};

        if (hasEmergency) {
            emergencyAlertBanner.className = "emergency-banner banner-green";
            bannerIcon.textContent = "🚨";
            bannerHeading.textContent = `EMERGENCY VEHICLE PREEMPTION ACTIVATED — ${data.emergency_road.toUpperCase()} APPROACH`;
            bannerReason.textContent = emAnalysis.reason || `Emergency confirmed — safely prioritizing ${data.emergency_road} direction.`;
            bannerBadge.textContent = "EMERGENCY PRIORITY CONFIRMED";
        } else if (sirenAnalysis.is_siren) {
            // Siren was detected, but no confirmed vehicle or ambiguous
            emergencyAlertBanner.className = "emergency-banner banner-amber";
            bannerIcon.textContent = "⚠️";
            bannerHeading.textContent = emAnalysis.is_ambiguous ? "AMBIGUOUS MULTI-DIRECTION EMERGENCY DETECTED" : "ACOUSTIC SIREN DETECTED WITHOUT VISUAL CONFIRMATION";
            bannerReason.textContent = emAnalysis.reason || "Siren detected — searching all directions for emergency vehicles.";
            bannerBadge.textContent = emAnalysis.is_ambiguous ? "AMBIGUOUS CANDIDATES" : "SEARCHING ALL DIRECTIONS";
        } else {
            // Normal adaptive mode
            emergencyAlertBanner.className = "emergency-banner banner-blue";
            bannerIcon.textContent = "ℹ️";
            bannerHeading.textContent = "Normal Adaptive Signal Optimization Active";
            bannerReason.textContent = emAnalysis.reason || "Normal adaptive mode: signal timing optimized based on vehicular queue.";
            bannerBadge.textContent = "NORMAL ADAPTIVE";
        }

        // 1.1 Render Dedicated Emergency Analysis & Dual-Verification Panel (Task 5)
        if (emergencyAnalysisPanel && data.emergency_analysis) {
            const ea = data.emergency_analysis;
            emergencyAnalysisPanel.style.display = "block";

            if (emAuditDirection) {
                emAuditDirection.innerHTML = `<span class="badge-dir">${ea.selected_direction || ea.emergency_road || "All Approaches (Auto-Scan)"}</span>`;
            }

            if (emAuditAmbulance) {
                const isAmb = Boolean(ea.ambulance_detected);
                emAuditAmbulance.innerHTML = `<span class="val-tag ${isAmb ? 'tag-yes' : 'tag-no'}">${isAmb ? 'Yes' : 'No'}</span>`;
            }

            if (emAuditFiretruck) {
                const isFt = Boolean(ea.fire_truck_detected);
                emAuditFiretruck.innerHTML = `<span class="val-tag ${isFt ? 'tag-yes' : 'tag-no'}">${isFt ? 'Yes' : 'No'}</span>`;
            }

            if (emAuditConfidence) {
                emAuditConfidence.textContent = (ea.emergency_confidence !== null && ea.emergency_confidence !== undefined)
                    ? `${(ea.emergency_confidence * 100).toFixed(1)}%`
                    : "--";
            }

            if (emAuditSiren) {
                const isSir = Boolean(ea.siren_detected);
                emAuditSiren.innerHTML = `<span class="val-tag ${isSir ? 'tag-yes' : 'tag-no'}">${isSir ? 'Yes' : 'No'}</span>`;
            }

            if (emAuditSirenConf) {
                emAuditSirenConf.textContent = (ea.siren_confidence !== null && ea.siren_confidence !== undefined)
                    ? `${(ea.siren_confidence * 100).toFixed(1)}%`
                    : "N/A";
            }

            if (emAuditConfirmation) {
                const isConf = Boolean(ea.confirmed);
                emAuditConfirmation.innerHTML = `<span class="val-tag ${isConf ? 'tag-confirmed' : 'tag-not-confirmed'}">${isConf ? 'Confirmed' : 'Not Confirmed'}</span>`;
            }

            if (emPanelStatusBadge) {
                const isConf = Boolean(ea.confirmed);
                emPanelStatusBadge.className = `val-tag ${isConf ? 'tag-confirmed' : 'tag-not-confirmed'}`;
                emPanelStatusBadge.textContent = isConf ? 'Confirmed' : 'Not Confirmed';
            }

            if (emAuditReason) {
                emAuditReason.textContent = ea.reason || "Normal adaptive mode.";
            }

            // Preview image
            if (emPreviewImg && emPreviewEmpty) {
                const previewSrc = ea.annotated_url || ea.original_url;
                if (previewSrc) {
                    emPreviewImg.src = previewSrc;
                    emPreviewImg.style.display = "block";
                    emPreviewEmpty.style.display = "none";
                } else {
                    emPreviewImg.src = "";
                    emPreviewImg.style.display = "none";
                    emPreviewEmpty.style.display = "flex";
                }
            }
        }

        // 2. Render 4 Result Direction Cards
        resultsCardsContainer.innerHTML = "";

        ROADS.forEach(roadKey => {
            const roadName = ROAD_NAMES[roadKey];
            const r = data.results[roadName];
            if (!r) return;

            const card = document.createElement("div");
            card.className = "result-card";

            // Status color class: green, blue, amber, red
            const statusClass = `status-${r.status_color || "blue"}`;
            const statusBadgeText = r.status_label || (r.emergency_active ? "Emergency Priority" : "Normal Adaptive");

            // Emergency vehicle detection label
            const hasEmergencyVehicle = Boolean(r.emergency_vehicle_detected || r.emergency_detected);
            const vType = (r.emergency_vehicle_type && r.emergency_vehicle_type !== "None") ? r.emergency_vehicle_type : null;
            let emVehicleDisplay = "None detected";
            let emValClass = "";

            if (hasEmergencyVehicle && vType) {
                const confPercent = (r.emergency_confidence !== null && r.emergency_confidence !== undefined)
                    ? ` (${Math.round(r.emergency_confidence * 100)}%)`
                    : "";

                if (r.emergency_active || r.emergency_priority_confirmed) {
                    emVehicleDisplay = `${vType} detected${confPercent}`;
                    emValClass = "em-found";
                } else {
                    // Vehicle detected by YOLO, but siren confirmation is pending/missing
                    emVehicleDisplay = `${vType} candidate — confirmation pending${confPercent}`;
                    emValClass = "em-pending";
                }
            } else {
                emVehicleDisplay = "None detected";
                emValClass = "";
            }

            // Density level badge color
            const densityLevel = (r.density && r.density.level) ? r.density.level : "LOW";
            const densityPercent = (r.density && r.density.percentage) ? r.density.percentage : 0;

            const counts = r.counts || { car: 0, bus: 0, truck: 0, motorcycle: 0, total: 0 };
            const maxGreen = r.max_green_time || 60;
            const finalGreen = r.prediction ? r.prediction.final_green_time : (r.green_time || 30);

            card.innerHTML = `
                <div class="result-card-header">
                    <div class="result-title-wrap">
                        <span class="road-compass-icon">${ROAD_ICONS[roadKey]}</span>
                        <span class="result-road-name">${roadName} Approach</span>
                    </div>
                    <span class="result-status-badge ${statusClass}">${statusBadgeText}</span>
                </div>

                <div class="result-media-wrap">
                    <img id="result-img-${roadKey}" src="${r.processed_url || r.original_url}" alt="${roadName} Annotated" class="result-thumbnail">
                    <div class="result-media-actions">
                        <button type="button" class="btn-sm btn-inspect" onclick="openLightbox('${r.processed_url || r.original_url}', '${roadName} Detection Annotations')">
                            🔍 Inspect Detection
                        </button>
                    </div>
                </div>

                <div class="result-metrics-grid">
                    <div class="metric-cell">
                        <span class="metric-cell-label">Total Vehicles</span>
                        <span class="metric-cell-value count-val">${counts.total}</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-cell-label">Traffic Density</span>
                        <span class="metric-cell-value density-${densityLevel.toLowerCase()}">${densityLevel} (${densityPercent}%)</span>
                    </div>
                    <div class="metric-cell">
                        <span class="metric-cell-label">Configured Max</span>
                        <span class="metric-cell-value">${maxGreen}s</span>
                    </div>
                    <div class="metric-cell metric-cell-green">
                        <span class="metric-cell-label">Allocated Green</span>
                        <span class="metric-cell-value highlight-green">${finalGreen}s</span>
                    </div>
                </div>

                <div class="result-vehicle-chips">
                    <span class="v-chip">🚗 Cars: ${counts.car}</span>
                    <span class="v-chip">🏍️ Bikes: ${counts.motorcycle}</span>
                    <span class="v-chip">🚌 Buses: ${counts.bus}</span>
                    <span class="v-chip">🚛 Trucks: ${counts.truck}</span>
                </div>

                <div class="result-emergency-info">
                    <span class="info-label">Emergency Vehicle:</span>
                    <span class="info-val ${emValClass}">${emVehicleDisplay}</span>
                </div>

                <div class="result-reason-msg">
                    ${r.status_message || "Normal adaptive mode: signal timing optimized based on vehicular queue."}
                </div>
            `;

            resultsCardsContainer.appendChild(card);
        });

        // 3. Render Cycle Breakdown Timeline
        const summaryCard = document.getElementById("cycle-summary-card");
        const summaryTitle = document.getElementById("cycle-summary-title");
        const timingNote = document.getElementById("timing-rule-note");

        if (cycleBarsContainer && data.cycle_plan && data.cycle_plan.sequence) {
            cycleBarsContainer.innerHTML = "";
            let totalSeconds = 0;

            const isEmergencyCycle = Boolean(data.emergency_active && data.emergency_road);

            if (summaryCard) {
                if (isEmergencyCycle) {
                    summaryCard.classList.add("emergency-priority-active");
                } else {
                    summaryCard.classList.remove("emergency-priority-active");
                }
            }

            if (summaryTitle) {
                if (isEmergencyCycle) {
                    summaryTitle.innerHTML = `🚦 Adaptive Cycle Allocation Summary — <span class="summary-priority-tag">🚨 EMERGENCY PRIORITY: ${data.emergency_road.toUpperCase()}</span>`;
                } else {
                    summaryTitle.innerHTML = `🚦 Adaptive Cycle Allocation Summary`;
                }
            }

            data.cycle_plan.sequence.forEach(step => {
                totalSeconds += step.green_time;
                const isStepPriority = Boolean(isEmergencyCycle && step.road === data.emergency_road);
                const bar = document.createElement("div");
                bar.className = `cycle-bar-segment ${isStepPriority ? "emergency-priority-bar" : ""}`;
                bar.style.flex = step.green_time;
                bar.innerHTML = `
                    <span class="bar-road">${isStepPriority ? '🚨 ' : ''}${step.road}${isStepPriority ? ' (PRIORITY)' : ''}</span>
                    <span class="bar-time">${step.green_time}s</span>
                `;
                cycleBarsContainer.appendChild(bar);
            });

            if (totalCycleBadge) {
                if (isEmergencyCycle) {
                    const emGreen = (data.green_times && data.green_times[data.emergency_road]) ? data.green_times[data.emergency_road] : 25;
                    totalCycleBadge.innerHTML = `Total Green: ${totalSeconds}s | <span class="badge-em-text">🚨 Priority: ${data.emergency_road} (${emGreen}s)</span>`;
                } else {
                    totalCycleBadge.textContent = `Total Cycle: ${totalSeconds}s (Max Safe Bounds Respected)`;
                }
            }

            if (timingNote) {
                if (isEmergencyCycle) {
                    const emGreen = (data.green_times && data.green_times[data.emergency_road]) ? data.green_times[data.emergency_road] : 25;
                    const gt = data.green_times || {};
                    timingNote.innerHTML = `
                        <strong>🚨 Emergency Priority Protocol Active:</strong> 
                        Conflicting approaches stopped (zero simultaneous green signals). 
                        <strong>${data.emergency_road} Approach</strong> allocated <strong>${emGreen}s</strong> priority green (respecting safety bounds [15s, 60s]). 
                        All directions updated: North (${gt.North !== undefined ? gt.North : '--'}s), East (${gt.East !== undefined ? gt.East : '--'}s), South (${gt.South !== undefined ? gt.South : '--'}s), West (${gt.West !== undefined ? gt.West : '--'}s). 
                        Resumes normal adaptive sequence automatically after clearance.
                    `;
                } else {
                    timingNote.innerHTML = `
                        <strong>Critical Timing Rule Enforced:</strong> 
                        <code>final_green_time = min(max_green_time, max(min_green_time, predicted_green_time))</code>. 
                        Recommended green duration never exceeds the configured maximum of 60s, respecting safety bounds [15s, 60s].
                    `;
                }
            }
        }

        // 4. Initialize Signal Simulator
        if (window.TrafficSimulator) {
            window.TrafficSimulator.init(data.cycle_plan, data.emergency_plan);
        }

        // Show results and simulation sections
        if (resultsSection) resultsSection.style.display = "block";
        if (simulationSection) simulationSection.style.display = "block";

        // Scroll smoothly to results
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    // --------------------------------------------------------------------------
    // Lightbox Modal Handling
    // --------------------------------------------------------------------------
    window.openLightbox = function (imgUrl, title) {
        if (!lightboxModal || !lightboxImg) return;
        lightboxImg.src = imgUrl;
        if (lightboxTitle) lightboxTitle.textContent = title || "Detection Details";
        lightboxModal.style.display = "flex";
    };

    function closeLightbox() {
        if (lightboxModal) lightboxModal.style.display = "none";
    }

})();
