/**
 * Advanced Client-Side 4-Way Traffic Signal Simulator & Telemetry Controller
 *
 * Implements:
 * 1. North -> East -> South -> West adaptive signal cycle.
 * 2. 3-Bulb Signal Controller: Red, Yellow (clearance), and Green states.
 * 3. Zero-conflict guarantee: Conflicting directions NEVER receive green simultaneously.
 * 4. Safe emergency preemption:
 *    - Conflicting green transitions to Yellow clearance (3s) -> Red.
 *    - All-red safety buffer (3s): ALL directions RED.
 *    - Emergency direction receives GREEN (25s).
 *    - Safe transition clearance (3s) -> Resumes normal adaptive cycle.
 * 5. Strict timer cleanup: No zombie callbacks or overlapping intervals on Stop/Restart.
 * 6. Dual-mode support: Works both in unified dashboard and standalone simulation.html.
 * 7. State management: Clean button enable/disable states, upcoming phase indicator,
 *    and real-time impact chart initialization.
 */

(function (global) {
    let timerId = null;
    let transitionTimeoutId = null;
    let emergencyTimerIds = [];
    
    let isRunning = false;
    let isPaused = false;
    let timeMultiplier = 5; // Default to 5x demo speed for fast interactive viewing

    let activeCyclePlan = null;
    let activeEmergencyPlan = null;

    let currentSequenceIndex = 0;
    let secondsRemaining = 0;
    let currentPhase = "idle"; // 'green', 'yellow', 'buffer', 'emergency_green', 'emergency_yellow'
    let currentActiveRoad = null;

    const ROADS = ["North", "East", "South", "West"];

    function clearAllTimers() {
        if (timerId) {
            clearInterval(timerId);
            timerId = null;
        }
        if (transitionTimeoutId) {
            clearTimeout(transitionTimeoutId);
            transitionTimeoutId = null;
        }
        emergencyTimerIds.forEach(id => {
            clearTimeout(id);
            clearInterval(id);
        });
        emergencyTimerIds = [];
    }

    function updateControlButtons() {
        const startBtn = document.getElementById("sim-btn-start") || document.getElementById("start-btn");
        const pauseBtn = document.getElementById("sim-btn-pause");
        const restartBtn = document.getElementById("sim-btn-restart") || document.getElementById("restart-btn");
        const statusText = document.getElementById("sim-live-status-text");

        if (!isRunning) {
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = '<span class="btn-icon">▶</span> Run Simulation';
            }
            if (pauseBtn) {
                pauseBtn.disabled = true;
                pauseBtn.innerHTML = '<span class="btn-icon">⏸</span> Pause';
            }
            if (restartBtn) restartBtn.disabled = true;
            if (statusText) statusText.textContent = "Status: Standby • Click Run Simulation to begin cycle";
        } else if (isPaused) {
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = '<span class="btn-icon">▶</span> Resume Simulation';
            }
            if (pauseBtn) {
                pauseBtn.disabled = true;
                pauseBtn.innerHTML = '<span class="btn-icon">⏸</span> Paused';
            }
            if (restartBtn) restartBtn.disabled = false;
            if (statusText) statusText.textContent = "Status: Paused • Click Resume Simulation to continue";
        } else {
            if (startBtn) {
                startBtn.disabled = true;
                startBtn.innerHTML = '<span class="btn-icon">▶</span> Running...';
            }
            if (pauseBtn) {
                pauseBtn.disabled = false;
                pauseBtn.innerHTML = '<span class="btn-icon">⏸</span> Pause';
            }
            if (restartBtn) restartBtn.disabled = false;
            if (statusText) {
                const roadDisplay = currentActiveRoad ? currentActiveRoad : "Cycle";
                statusText.textContent = `Status: Active • Phase: ${roadDisplay} (${secondsRemaining}s remaining)`;
            }
        }
    }

    function setAllBulbsRed() {
        ROADS.forEach(road => {
            const redBulb = document.getElementById(`bulb-red-${road}`) || document.getElementById(`light-${road}`);
            const yellowBulb = document.getElementById(`bulb-yellow-${road}`);
            const greenBulb = document.getElementById(`bulb-green-${road}`);

            if (redBulb) {
                redBulb.classList.add("active");
                redBulb.classList.add("red");
                redBulb.classList.remove("green");
            }
            if (yellowBulb) yellowBulb.classList.remove("active");
            if (greenBulb) greenBulb.classList.remove("active");

            const timerBadge = document.getElementById(`sim-timer-${road}`) || document.getElementById(`time-${road}`);
            if (timerBadge) {
                timerBadge.classList.remove("active-green");
            }

            const approachLane = document.getElementById(`approach-${road}`);
            if (approachLane) {
                approachLane.classList.remove("active-green-approach", "active-yellow-approach");
            }
        });
    }

    function setRoadSignal(road, state) {
        // state: 'green' | 'yellow' | 'red'
        // First, ensure all conflicting directions are strictly locked RED
        ROADS.forEach(r => {
            if (r !== road) {
                const redBulb = document.getElementById(`bulb-red-${r}`) || document.getElementById(`light-${r}`);
                const yellowBulb = document.getElementById(`bulb-yellow-${r}`);
                const greenBulb = document.getElementById(`bulb-green-${r}`);

                if (redBulb) {
                    redBulb.classList.add("active");
                    redBulb.classList.add("red");
                    redBulb.classList.remove("green");
                }
                if (yellowBulb) yellowBulb.classList.remove("active");
                if (greenBulb) greenBulb.classList.remove("active");

                const badge = document.getElementById(`sim-timer-${r}`) || document.getElementById(`time-${r}`);
                if (badge) badge.classList.remove("active-green");

                const lane = document.getElementById(`approach-${r}`);
                if (lane) lane.classList.remove("active-green-approach", "active-yellow-approach");
            }
        });

        // Set the active road's bulbs
        const activeRed = document.getElementById(`bulb-red-${road}`) || document.getElementById(`light-${road}`);
        const activeYellow = document.getElementById(`bulb-yellow-${road}`);
        const activeGreen = document.getElementById(`bulb-green-${road}`);
        const timerBadge = document.getElementById(`sim-timer-${road}`) || document.getElementById(`time-${road}`);
        const activeLane = document.getElementById(`approach-${road}`);

        if (activeRed) {
            if (state === "red") {
                activeRed.classList.add("active");
                activeRed.classList.add("red");
                activeRed.classList.remove("green");
            } else {
                activeRed.classList.remove("active");
                activeRed.classList.remove("red");
            }
        }

        if (activeYellow) {
            if (state === "yellow") {
                activeYellow.classList.add("active");
            } else {
                activeYellow.classList.remove("active");
            }
        }

        if (activeGreen) {
            if (state === "green") {
                activeGreen.classList.add("active");
                if (activeRed) activeRed.classList.add("green"); // for legacy single-light element
            } else {
                activeGreen.classList.remove("active");
            }
        }

        if (timerBadge) {
            if (state === "green") {
                timerBadge.classList.add("active-green");
            } else {
                timerBadge.classList.remove("active-green");
            }
        }

        if (activeLane) {
            activeLane.classList.remove("active-green-approach", "active-yellow-approach");
            if (state === "green") {
                activeLane.classList.add("active-green-approach");
            } else if (state === "yellow") {
                activeLane.classList.add("active-yellow-approach");
            }
        }
    }

    function updateUpcomingPhase() {
        const upcomingEl = document.getElementById("sim-upcoming-phase");
        if (!upcomingEl) return;

        if (!isRunning) {
            if (activeEmergencyPlan && activeEmergencyPlan.emergency_road) {
                upcomingEl.textContent = `Upcoming: Safety Buffer (3s) ➔ ${activeEmergencyPlan.emergency_road} (Emergency Corridor)`;
            } else if (activeCyclePlan && activeCyclePlan.sequence && activeCyclePlan.sequence.length > 0) {
                const first = activeCyclePlan.sequence[0];
                upcomingEl.textContent = `Planned 1st: ${first.road} Approach (${first.green_time}s)`;
            } else {
                upcomingEl.textContent = "Planned: Ready to start simulation";
            }
            return;
        }

        if (currentPhase === "green") {
            if (activeCyclePlan && activeCyclePlan.sequence) {
                const nextIndex = (currentSequenceIndex + 1) % activeCyclePlan.sequence.length;
                const nextStep = activeCyclePlan.sequence[nextIndex];
                upcomingEl.textContent = `Next Phase: ${nextStep.road} Approach (Planned: ${nextStep.green_time}s)`;
            }
        } else if (currentPhase === "yellow") {
            if (activeCyclePlan && activeCyclePlan.sequence) {
                const nextIndex = (currentSequenceIndex + 1) % activeCyclePlan.sequence.length;
                const nextStep = activeCyclePlan.sequence[nextIndex];
                upcomingEl.textContent = `Yellow Clearance Active ➔ Advancing to ${nextStep.road} in ${secondsRemaining}s`;
            }
        } else if (currentPhase === "buffer") {
            const emRoad = activeEmergencyPlan ? activeEmergencyPlan.emergency_road : "Emergency Road";
            upcomingEl.textContent = `All-Red Safety Buffer ➔ Granting Priority to ${emRoad} in ${secondsRemaining}s`;
        } else if (currentPhase === "emergency_green") {
            upcomingEl.textContent = `Emergency Priority Corridor Active ➔ Clearance in ${secondsRemaining}s`;
        } else if (currentPhase === "emergency_yellow") {
            upcomingEl.textContent = `Emergency Clearance Complete ➔ Resuming Normal Adaptive Cycle (North)`;
        }
    }

    function updateCenterDisplay(roadName, countdownSeconds, phaseLabel, phaseClass = "") {
        const roadEl = document.getElementById("sim-active-road") || document.getElementById("sim-center-road") || document.getElementById("current-road");
        const countEl = document.getElementById("sim-countdown") || document.getElementById("sim-center-countdown") || document.getElementById("countdown");
        const phaseEl = document.getElementById("sim-phase-badge") || document.getElementById("sim-center-phase");

        if (roadEl) roadEl.textContent = roadName;
        if (countEl) {
            countEl.textContent = countdownSeconds >= 0 ? countdownSeconds : 0;
            countEl.className = `center-countdown ${phaseClass}`;
        }
        if (phaseEl) phaseEl.textContent = phaseLabel;

        // Also update per-road timer badge
        if (roadName && roadName !== "-" && !roadName.includes("BUFFER") && !roadName.includes("STANDBY")) {
            const badge = document.getElementById(`sim-timer-${roadName}`) || document.getElementById(`time-${roadName}`);
            if (badge) badge.textContent = `${countdownSeconds}s`;
        }

        updateUpcomingPhase();
        updateControlButtons();
    }

    function updateModeBanner(text, isEmergency) {
        const banner = document.getElementById("sim-mode-banner") || document.getElementById("mode-banner");
        if (banner) {
            banner.textContent = text;
            banner.className = `mode-banner-strip ${isEmergency ? "emergency" : "normal"}`;
        }
    }

    function tick() {
        if (!isRunning || isPaused) return;

        secondsRemaining -= 1;
        
        let phaseLabel = "GREEN PHASE";
        let phaseClass = "";
        if (currentPhase === "yellow") {
            phaseLabel = "YELLOW CLEARANCE";
            phaseClass = "yellow-phase";
        } else if (currentPhase === "buffer") {
            phaseLabel = "ALL-RED BUFFER";
            phaseClass = "buffer-phase";
        } else if (currentPhase === "emergency_green") {
            phaseLabel = "EMERGENCY PREEMPTION";
            phaseClass = "";
        }

        updateCenterDisplay(currentActiveRoad, secondsRemaining, phaseLabel, phaseClass);

        if (secondsRemaining <= 0) {
            advancePhase();
        }
    }

    function advancePhase() {
        if (!activeCyclePlan || !activeCyclePlan.sequence || activeCyclePlan.sequence.length === 0) {
            stopSimulation();
            return;
        }

        const yellowDuration = activeCyclePlan.yellow_duration || 3;

        if (currentPhase === "green") {
            // Transition Green -> Yellow (safe clearance)
            currentPhase = "yellow";
            secondsRemaining = yellowDuration;
            setRoadSignal(currentActiveRoad, "yellow");
            updateCenterDisplay(currentActiveRoad, secondsRemaining, "YELLOW CLEARANCE", "yellow-phase");
        } else if (currentPhase === "yellow") {
            // Yellow completed -> Advance to next road in sequence
            currentSequenceIndex = (currentSequenceIndex + 1) % activeCyclePlan.sequence.length;
            const nextStep = activeCyclePlan.sequence[currentSequenceIndex];
            startRoadPhase(nextStep.road, nextStep.green_time);
        } else if (currentPhase === "buffer") {
            // Buffer completed during emergency transition -> Grant Green to Emergency Road
            const emRoad = activeEmergencyPlan ? activeEmergencyPlan.emergency_road : "South";
            const emDuration = activeEmergencyPlan ? activeEmergencyPlan.emergency_green_seconds : 25;
            startEmergencyGreen(emRoad, emDuration);
        } else if (currentPhase === "emergency_green") {
            // Emergency green completed -> Transition Yellow clearance -> Resume normal cycle
            currentPhase = "emergency_yellow";
            secondsRemaining = yellowDuration;
            setRoadSignal(currentActiveRoad, "yellow");
            updateCenterDisplay(currentActiveRoad, secondsRemaining, "YELLOW CLEARANCE", "yellow-phase");
        } else if (currentPhase === "emergency_yellow") {
            // Emergency completely cleared -> Resume normal adaptive cycle from North
            updateModeBanner("NORMAL ADAPTIVE MODE (Resumed)", false);
            activeEmergencyPlan = null; // Clear so subsequent cycles run normal
            currentSequenceIndex = 0;
            const firstStep = activeCyclePlan.sequence[0];
            startRoadPhase(firstStep.road, firstStep.green_time);
        }
    }

    function startRoadPhase(road, duration) {
        currentPhase = "green";
        currentActiveRoad = road;
        secondsRemaining = duration;
        setRoadSignal(road, "green");
        updateCenterDisplay(road, duration, "GREEN PHASE");
    }

    function startEmergencyGreen(road, duration) {
        currentPhase = "emergency_green";
        currentActiveRoad = road;
        secondsRemaining = duration;
        setRoadSignal(road, "green");
        updateCenterDisplay(road, duration, "EMERGENCY PREEMPTION");
    }

    function runEmergencyTransition() {
        const emRoad = activeEmergencyPlan.emergency_road;
        const vehicleType = activeEmergencyPlan.vehicle_type || "Emergency Vehicle";
        const bufferStep = (activeEmergencyPlan.steps || []).find(s => s.phase === "safety_buffer");
        const bufferDuration = bufferStep ? bufferStep.duration : 3;

        updateModeBanner(`🚨 EMERGENCY PRIORITY ACTIVATED — ${vehicleType} on ${emRoad}`, true);

        // Step 1: All lights to RED for safety buffer
        setAllBulbsRed();
        currentPhase = "buffer";
        currentActiveRoad = "ALL-RED BUFFER";
        secondsRemaining = bufferDuration;
        updateCenterDisplay("ALL-RED BUFFER", bufferDuration, "ALL-RED BUFFER", "buffer-phase");
    }

    function startSimulation() {
        if (isRunning && !isPaused) return;

        if (isPaused) {
            isPaused = false;
            updateControlButtons();
            timerId = setInterval(tick, 1000 / timeMultiplier);
            return;
        }

        clearAllTimers();
        isRunning = true;
        isPaused = false;

        // Reset display
        setAllBulbsRed();
        updateControlButtons();

        if (activeEmergencyPlan && activeEmergencyPlan.emergency_road) {
            runEmergencyTransition();
        } else {
            updateModeBanner("NORMAL ADAPTIVE MODE", false);
            currentSequenceIndex = 0;
            const firstStep = activeCyclePlan.sequence[0];
            startRoadPhase(firstStep.road, firstStep.green_time);
        }

        timerId = setInterval(tick, 1000 / timeMultiplier);
    }

    function pauseSimulation() {
        if (!isRunning) return;
        isPaused = true;
        if (timerId) {
            clearInterval(timerId);
            timerId = null;
        }
        updateControlButtons();
        const phaseEl = document.getElementById("sim-phase-badge") || document.getElementById("sim-center-phase");
        if (phaseEl) phaseEl.textContent = "SIMULATION PAUSED";
    }

    function stopSimulation() {
        isRunning = false;
        isPaused = false;
        clearAllTimers();
        setAllBulbsRed();

        updateCenterDisplay("STANDBY", "--", "READY TO START");
        updateModeBanner(activeEmergencyPlan ? "EMERGENCY PREEMPTION STANDBY" : "NORMAL ADAPTIVE CYCLE", Boolean(activeEmergencyPlan));
        updateControlButtons();
    }

    function restartSimulation() {
        stopSimulation();
        startSimulation();
    }

    function setSpeed(multiplier) {
        timeMultiplier = Number(multiplier) || 1;
        if (isRunning && !isPaused) {
            clearInterval(timerId);
            timerId = setInterval(tick, 1000 / timeMultiplier);
        }
    }

    function initSimulation(cyclePlan, emergencyPlan) {
        activeCyclePlan = cyclePlan;
        activeEmergencyPlan = emergencyPlan;
        stopSimulation();

        // Populate initial time indicators from sequence
        if (activeCyclePlan && activeCyclePlan.sequence) {
            activeCyclePlan.sequence.forEach(step => {
                const badge = document.getElementById(`sim-timer-${step.road}`) || document.getElementById(`time-${step.road}`);
                if (badge) badge.textContent = `${step.green_time}s`;
            });
        }
    }

    function initImpactCharts() {
        if (typeof Plotly === "undefined" || !global.IMPACT_ASSESSMENT) return;
        const ia = global.IMPACT_ASSESSMENT;
        const opt = ia.optimization_results;
        const sys = ia.system_impact;
        if (!opt || !sys) return;

        // 1. Green Time Reallocation Chart
        const chartGreenEl = document.getElementById("chart-green-reallocation");
        if (chartGreenEl) {
            const roads = ["North", "East", "South", "West"];
            const baselineTimes = roads.map(r => opt.approaches[r] ? opt.approaches[r].baseline_green : 30);
            const allocatedTimes = roads.map(r => opt.approaches[r] ? opt.approaches[r].allocated_green : 30);

            const trace1 = {
                x: roads,
                y: baselineTimes,
                name: "Baseline Fixed (30s)",
                type: "bar",
                marker: { color: "#475569" },
            };

            const trace2 = {
                x: roads,
                y: allocatedTimes,
                name: "AI-Adaptive Allocated",
                type: "bar",
                marker: { color: "#10b981" },
            };

            const layout1 = {
                barmode: "group",
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#cbd5e1", family: "Inter, sans-serif", size: 11 },
                margin: { l: 45, r: 15, t: 25, b: 35 },
                legend: { orientation: "h", y: 1.15, x: 0 },
                yaxis: {
                    title: "Green Seconds",
                    gridcolor: "rgba(255,255,255,0.08)",
                    zerolinecolor: "rgba(255,255,255,0.15)",
                },
                xaxis: {
                    gridcolor: "transparent",
                },
            };

            Plotly.newPlot(chartGreenEl, [trace1, trace2], layout1, { responsive: true, displayModeBar: false });
        }

        // 2. Throughput & Queue Comparison Chart
        const chartQueueEl = document.getElementById("chart-throughput-queue");
        if (chartQueueEl) {
            const categories = ["Baseline (Fixed)", "AI-Adaptive"];
            const served = [
                sys.vehicles_processed.baseline_served_vehicles,
                sys.vehicles_processed.optimized_served_vehicles,
            ];
            const unserved = [
                sys.queue_length.baseline_unserved_vehicles,
                sys.queue_length.optimized_unserved_vehicles,
            ];

            const traceServed = {
                x: categories,
                y: served,
                name: "Vehicles Processed (Served)",
                type: "bar",
                marker: { color: "#3b82f6" },
            };

            const traceQueue = {
                x: categories,
                y: unserved,
                name: "Unserved Residual Queue",
                type: "bar",
                marker: { color: "#f59e0b" },
            };

            const layout2 = {
                barmode: "group",
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#cbd5e1", family: "Inter, sans-serif", size: 11 },
                margin: { l: 45, r: 15, t: 25, b: 35 },
                legend: { orientation: "h", y: 1.15, x: 0 },
                yaxis: {
                    title: "Vehicles",
                    gridcolor: "rgba(255,255,255,0.08)",
                    zerolinecolor: "rgba(255,255,255,0.15)",
                },
                xaxis: {
                    gridcolor: "transparent",
                },
            };

            Plotly.newPlot(chartQueueEl, [traceServed, traceQueue], layout2, { responsive: true, displayModeBar: false });
        }
    }

    // Attach UI event listeners if elements exist
    document.addEventListener("DOMContentLoaded", () => {
        const startBtn = document.getElementById("sim-btn-start") || document.getElementById("start-btn");
        const pauseBtn = document.getElementById("sim-btn-pause");
        const stopBtn = document.getElementById("sim-btn-stop") || document.getElementById("stop-btn");
        const restartBtn = document.getElementById("sim-btn-restart") || document.getElementById("restart-btn");
        const speedSelect = document.getElementById("sim-speed-select");

        if (startBtn) startBtn.addEventListener("click", startSimulation);
        if (pauseBtn) pauseBtn.addEventListener("click", pauseSimulation);
        if (stopBtn) stopBtn.addEventListener("click", stopSimulation);
        if (restartBtn) restartBtn.addEventListener("click", restartSimulation);
        if (speedSelect) {
            speedSelect.addEventListener("change", (e) => setSpeed(e.target.value));
            setSpeed(speedSelect.value);
        }

        // Keyboard navigation shortcuts
        document.addEventListener("keydown", (e) => {
            if (e.target && ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
            if (e.code === "Space") {
                e.preventDefault();
                if (isRunning && !isPaused) pauseSimulation();
                else startSimulation();
            } else if (e.code === "KeyR") {
                restartSimulation();
            }
        });

        // Initialize from global window variables if rendered via template
        if (typeof global.CYCLE_PLAN !== "undefined" && global.CYCLE_PLAN !== null) {
            initSimulation(global.CYCLE_PLAN, global.EMERGENCY_PLAN || null);
        }

        // Initialize impact telemetry charts
        initImpactCharts();
    });

    // Expose controller API to global window
    global.TrafficSimulator = {
        init: initSimulation,
        start: startSimulation,
        pause: pauseSimulation,
        stop: stopSimulation,
        restart: restartSimulation,
        setSpeed: setSpeed,
        clearAllTimers: clearAllTimers,
        isRunning: () => isRunning,
        getCurrentState: () => ({
            road: currentActiveRoad,
            phase: currentPhase,
            remaining: secondsRemaining,
            emergency: Boolean(activeEmergencyPlan),
        }),
    };
})(window);
