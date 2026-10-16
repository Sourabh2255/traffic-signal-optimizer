/**
 * Advanced Client-Side 4-Way Traffic Signal Simulator
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
 */

(function (global) {
    let timerId = null;
    let transitionTimeoutId = null;
    let emergencyTimerIds = [];
    
    let isRunning = false;
    let isPaused = false;
    let timeMultiplier = 1; // 1x, 2x, 5x simulation speed

    let activeCyclePlan = null;
    let activeEmergencyPlan = null;

    let currentSequenceIndex = 0;
    let secondsRemaining = 0;
    let currentPhase = "idle"; // 'green', 'yellow', 'buffer', 'emergency_green'
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
        });
    }

    function setRoadSignal(road, state) {
        // state: 'green' | 'yellow' | 'red'
        // First, ensure all conflicting directions are RED
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
            }
        });

        // Set the active road's bulbs
        const activeRed = document.getElementById(`bulb-red-${road}`) || document.getElementById(`light-${road}`);
        const activeYellow = document.getElementById(`bulb-yellow-${road}`);
        const activeGreen = document.getElementById(`bulb-green-${road}`);
        const timerBadge = document.getElementById(`sim-timer-${road}`) || document.getElementById(`time-${road}`);

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
    }

    function updateCenterDisplay(roadName, countdownSeconds, phaseLabel, phaseClass = "") {
        const roadEl = document.getElementById("sim-center-road") || document.getElementById("current-road");
        const countEl = document.getElementById("sim-center-countdown") || document.getElementById("countdown");
        const phaseEl = document.getElementById("sim-center-phase");

        if (roadEl) roadEl.textContent = roadName;
        if (countEl) {
            countEl.textContent = countdownSeconds >= 0 ? countdownSeconds : 0;
            countEl.className = `center-countdown ${phaseClass}`;
        }
        if (phaseEl) phaseEl.textContent = phaseLabel;

        // Also update per-road timer badge
        if (roadName && roadName !== "-" && !roadName.includes("BUFFER")) {
            const badge = document.getElementById(`sim-timer-${roadName}`) || document.getElementById(`time-${roadName}`);
            if (badge) badge.textContent = `${countdownSeconds}s`;
        }
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
        currentActiveRoad = "SAFETY BUFFER (ALL RED)";
        secondsRemaining = bufferDuration;
        updateCenterDisplay("SAFETY BUFFER (ALL RED)", bufferDuration, "ALL-RED BUFFER", "buffer-phase");
    }

    function startSimulation() {
        if (isRunning && !isPaused) return;

        if (isPaused) {
            isPaused = false;
            timerId = setInterval(tick, 1000 / timeMultiplier);
            return;
        }

        clearAllTimers();
        isRunning = true;
        isPaused = false;

        // Reset display
        setAllBulbsRed();

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
        const phaseEl = document.getElementById("sim-center-phase");
        if (phaseEl) phaseEl.textContent = "SIMULATION PAUSED";
    }

    function stopSimulation() {
        isRunning = false;
        isPaused = false;
        clearAllTimers();
        setAllBulbsRed();

        updateCenterDisplay("-", "--", "READY TO START");
        updateModeBanner("NORMAL ADAPTIVE MODE", false);
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
        }

        // Initialize from global window variables if rendered via template
        if (typeof global.CYCLE_PLAN !== "undefined") {
            initSimulation(global.CYCLE_PLAN, global.EMERGENCY_PLAN || null);
        }
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
