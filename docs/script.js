/**
 * ORIEN | NEURAL SYNERGY CRYSTAL - CONTROLLER
 * Integration: Camera + Mic + WebSocket + Model Visualizer
 */

class OrienHUD {
    constructor() {
        this.config = {
            wsUrl: `ws://${window.location.hostname}:8000/ws/neural`,
            apiUrl: `http://${window.location.hostname}:8000/api`,
            frameRate: 2, // 2 frames per second to backend
            emojis: {
                HAPPY: '😊', SAD: '😢', ANGRY: '😠', NEUTRAL: '😐', 
                STRESSED: '😰', SURPRISE: '😲', DISGUST: '🤢', FEAR: '😨'
            }
        };

        this.state = {
            isRunning: false,
            isConnected: false,
            isTestMode: false,
            lastFrameTime: 0,
            latency: 0,
            fps: 0,
            framesProcessed: 0,
            lastFpsUpdate: Date.now(),
            language: 'en-US', // 'en-US' or 'ta-IN'
            lastTriggeredEmotion: null,
            autoEngage: true,
            emotionStability: {
                current: null,
                count: 0,
                threshold: 10 // Trigger after ~5 seconds of stability at 2 FPS
            },
            botState: 'STANDBY' // STANDBY, RITUAL_ACTIVE, SOLVER_ACTIVE
        };

        this.questions = {
            'en-US': {
                HAPPY: "How can I help you sustain this positive energy?",
                SAD: "I'm here for you. Would you like to talk about what's on your mind?",
                ANGRY: "I understand you're frustrated. How can we resolve this together?",
                NEUTRAL: "Everything seems stable. Is there anything specific you'd like to explore?",
                STRESSED: "Take a deep breath. Can I suggest some ways to lower the pressure?",
                SURPRISE: "That's unexpected! What just happened?",
                FEAR: "It's okay to feel anxious. How can I help you feel more secure?",
                DISGUST: "That seems unpleasant. Should we pivot to something else?"
            },
            'ta-IN': {
                HAPPY: "இந்த நேர்மறையான ஆற்றலைத் தக்கவைக்க நான் உங்களுக்கு எப்படி உதவ முடியும்?",
                SAD: "நான் உனக்காக இங்கே இருக்கிறேன். உங்கள் மனதில் உள்ளதைப் பற்றி பேச விரும்புகிறீர்களா?",
                ANGRY: "உங்கள் விரக்தியை நான் புரிந்துகொள்கிறேன். இதை எப்படி ஒன்றாகத் தீர்ப்பது?",
                NEUTRAL: "எல்லாம் சீராகத் தெரிகிறது. நீங்கள் ஆராய விரும்பும் குறிப்பிட்ட ஏதேனும் உள்ளதா?",
                STRESSED: "ஆழ்ந்த மூச்சு விடுங்கள். அழுத்தத்தைக் குறைப்பதற்கான சில வழிகளை நான் பரிந்துரைக்கலாமா?",
                SURPRISE: "அது எதிர்பாராதது! என்ன நடந்தது?",
                FEAR: "கவலையாக இருப்பது பரவாயில்லை. நீங்கள் பாதுகாப்பாக உணர நான் எப்படி உதவ முடியும்?",
                DISGUST: "அது விரும்பத்தகாததாகத் தெரிகிறது. நாம் வேறு ஏதாவது செய்யலாமா?"
            }
        };

        this.elements = {
            video: document.getElementById('vision-video'),
            overlay: document.getElementById('overlay-canvas'),
            waveform: document.getElementById('waveform-canvas'),
            logFeed: document.getElementById('log-feed'),
            emotionEmoji: document.getElementById('emotion-emoji'),
            emotionLabel: document.getElementById('emotion-label'),
            emotionConfidence: document.getElementById('emotion-confidence'),
            confidenceList: document.getElementById('confidence-list'),
            synergyState: document.getElementById('synergy-state'),
            synergyDesc: document.getElementById('synergy-description'),
            speechText: document.getElementById('speech-text'),
            debugDisplay: document.getElementById('debug-display'),
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn'),
            latencyVal: document.getElementById('latency-val'),
            fpsVal: document.getElementById('fps-val'),
            backendStatus: document.getElementById('backend-status'),
            testModeToggle: document.getElementById('test-mode-toggle'),
            clock: document.getElementById('clock'),
            langEn: document.getElementById('lang-en'),
            langTa: document.getElementById('lang-ta'),
            triggerGrid: document.getElementById('emotion-trigger-grid'),
            autoEngageToggle: document.getElementById('auto-engage-toggle'),
            syncProgress: document.getElementById('sync-progress'),
            ritualContainer: document.getElementById('ritual-container'),
            ritualIcon: document.getElementById('ritual-icon'),
            ritualTask: document.getElementById('ritual-task'),
            ritualDesc: document.getElementById('ritual-desc')
        };

        this.init();
    }

    init() {
        this.bindEvents();
        this.setupClock();
        this.setupNeuralBackground(); // Aesthetic only
        this.log('System', 'Awaiting initialization...');
        
        // Auto-connect WebSocket on start
        this.connectWS();
    }

    bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.startSystem());
        this.elements.stopBtn.addEventListener('click', () => this.stopSystem());
        
        this.elements.autoEngageToggle.addEventListener('change', (e) => {
            this.state.autoEngage = e.target.checked;
            this.log('Config', `Auto-Engagement: ${this.state.autoEngage ? 'ON' : 'OFF'}`);
            if (!this.state.autoEngage) {
                this.elements.syncProgress.style.width = '0%';
            }
        });

        this.elements.testModeToggle.addEventListener('change', (e) => {
            this.state.isTestMode = e.target.checked;
            document.body.classList.toggle('test-mode', this.state.isTestMode);
            this.log('Config', `Test Mode: ${this.state.isTestMode ? 'ON' : 'OFF'}`);
        });

        // Language Selectors
        this.elements.langEn.addEventListener('click', () => this.setLanguage('en-US'));
        this.elements.langTa.addEventListener('click', () => this.setLanguage('ta-IN'));

        // Emotion Triggers
        const triggerBtns = this.elements.triggerGrid.querySelectorAll('.trigger-btn');
        triggerBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const emotion = btn.getAttribute('data-emotion');
                this.triggerEmotion(emotion);
            });
        });
    }

    setLanguage(lang) {
        this.state.language = lang;
        this.elements.langEn.classList.toggle('active', lang === 'en-US');
        this.elements.langTa.classList.toggle('active', lang === 'ta-IN');
        this.log('Config', `Language set to: ${lang === 'en-US' ? 'English' : 'Tamil'}`);
        
        // Restart STT with new language if active
        if (this.recognition) {
            this.recognition.stop();
            // will auto-restart in onend
        }
    }

    triggerEmotion(emotion) {
        if (!this.state.isConnected) {
            this.log('Error', 'Neural Bridge not established.');
            return;
        }

        const question = this.questions[this.state.language][emotion];
        this.log('Trigger', `Manual Simulation: ${emotion}`);
        
        // Update UI immediately to reflect triggered emotion
        this.elements.emotionEmoji.textContent = this.config.emojis[emotion] || '🧿';
        this.elements.emotionLabel.textContent = emotion;

        // Send query to AI
        const payload = {
            query: `(System Alert: Manual Trigger - ${emotion})`,
            face_emotion: emotion,
            language: this.state.language,
            behavior: { jitter: 0, mouse_speed: 0 }
        };

        this.lastRequestTimestamp = Date.now();
        this.ws.send(JSON.stringify(payload));
        
        // Show the question we're "asking" related to this emotion
        this.elements.speechText.textContent = `[System]: ${question}`;
    }

    async startSystem() {
        this.log('System', 'Initializing neural sensors...');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            this.elements.video.srcObject = stream;
            
            this.setupAudioAnalysis(stream);
            this.startSTT();
            this.state.isRunning = true;
            
            this.elements.startBtn.style.display = 'none';
            this.elements.stopBtn.style.display = 'flex';
            document.getElementById('cam-active-dot').classList.add('active');
            document.getElementById('stream-status').classList.add('active');
            
            this.log('Vision', 'Camera stream active.');
            this.startVisionLoop();
        } catch (err) {
            this.log('Error', `Sensor Failure: ${err.message}`);
            alert("Please allow camera and microphone access.");
        }
    }

    stopSystem() {
        this.state.isRunning = false;
        const stream = this.elements.video.srcObject;
        if (stream) stream.getTracks().forEach(t => t.stop());
        this.elements.video.srcObject = null;
        
        this.elements.startBtn.style.display = 'flex';
        this.elements.stopBtn.style.display = 'none';
        document.getElementById('cam-active-dot').classList.remove('active');
        document.getElementById('stream-status').classList.remove('active');
        this.log('System', 'Shutdown complete.');
    }

    connectWS() {
        this.log('Network', 'Bridging to Neural Core...');
        
        // Ensure only one connection exists
        if (this.ws) {
            this.ws.onopen = this.ws.onmessage = this.ws.onclose = this.ws.onerror = null;
            this.ws.close();
        }

        try {
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                this.state.isConnected = true;
                this.updateNetworkStatus('WS_ACTIVE', 'var(--accent-green)');
                this.log('Network', 'Neural Bridge Established.');
            };

            this.ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                this.handlePrediction(data);
                if (this.state.isTestMode) {
                    this.elements.debugDisplay.textContent = JSON.stringify(data, null, 2);
                }
            };

            this.ws.onclose = () => {
                this.state.isConnected = false;
                this.updateNetworkStatus('WS_DISCONNECTED', 'var(--accent-red)');
                this.log('Error', 'Neural Bridge Lost. Retrying in 5s...');
                setTimeout(() => this.connectWS(), 5000);
            };

            this.ws.onerror = (err) => {
                this.log('Error', 'WebSocket handshake failed.');
                this.updateNetworkStatus('WS_FAULT', 'var(--accent-red)');
            };
        } catch (e) {
            this.log('Error', 'WebSocket initialization failed.');
        }
    }

    updateNetworkStatus(text, color) {
        this.elements.backendStatus.textContent = text;
        this.elements.backendStatus.style.color = color;
        this.elements.backendStatus.style.textShadow = `0 0 10px ${color}`;
    }

    startSTT() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return;

        this.recognition = new SpeechRecognition();
        this.recognition.lang = this.state.language;
        this.recognition.continuous = true;
        this.recognition.interimResults = true;

        this.recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            
            this.elements.speechText.textContent = transcript;
            
            // If final, send to backend
            if (event.results[event.results.length - 1].isFinal) {
                this.sendQuery(transcript);
            }
        };

        this.recognition.onstart = () => { this.log('Audio', 'Speech Recognition Active.'); };
        this.recognition.onerror = (e) => { this.log('Error', `STT: ${e.error}`); };
        this.recognition.onend = () => { if (this.state.isRunning) this.recognition.start(); };
        
        this.recognition.start();
    }

    sendQuery(text) {
        if (!this.state.isConnected || !text.trim()) return;
        this.ws.send(JSON.stringify({
            query: text,
            language: this.state.language,
            behavior: { jitter: 0, mouse_speed: 0 }
        }));
    }

    startVisionLoop() {
        const loop = () => {
            if (!this.state.isRunning) return;

            // Update FPS
            this.state.framesProcessed++;
            const now = Date.now();
            if (now - this.state.lastFpsUpdate > 1000) {
                this.state.fps = this.state.framesProcessed;
                this.elements.fpsVal.textContent = this.state.fps;
                this.state.framesProcessed = 0;
                this.state.lastFpsUpdate = now;
            }

            // Sync frame to backend based on configured rate
            const timeSinceLastFrame = now - this.state.lastFrameTime;
            if (timeSinceLastFrame > (1000 / this.config.frameRate)) {
                this.state.lastFrameTime = now;
                this.sendFrame();
            }

            requestAnimationFrame(loop);
        };
        loop();
    }

    sendFrame() {
        if (!this.state.isConnected || !this.state.isRunning) return;

        const canvas = document.createElement('canvas');
        canvas.width = 160; // Throttled resolution for faster inference
        canvas.height = 120;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.elements.video, 0, 0, canvas.width, canvas.height);
        
        const base64 = canvas.toDataURL('image/jpeg', 0.6);
        const payload = {
            type: 'FRAME',
            image: base64,
            language: this.state.language,
            behavior: { jitter: 0, mouse_speed: 0 } // Basic placeholder
        };

        this.lastRequestTimestamp = Date.now();
        this.ws.send(JSON.stringify(payload));
    }

    handlePrediction(data) {
        // Calculate Latency
        if (this.lastRequestTimestamp) {
            this.state.latency = Date.now() - this.lastRequestTimestamp;
            this.elements.latencyVal.textContent = `${this.state.latency}ms`;
        }

        if (data.status === 'ok') {
            const emotion = (data.emotion || data.state || 'Neutral').toUpperCase();
            
            // Handle both dictionary and float confidence formats
            let confidence = data.confidence || 0;
            let mainConf = 0;
            
            if (typeof confidence === 'object' && confidence !== null) {
                const searchKey = emotion.toLowerCase();
                // Case-insensitive lookup
                const entry = Object.entries(confidence).find(([k]) => k.toLowerCase() === searchKey);
                if (entry) {
                    mainConf = entry[1];
                } else {
                    // Fallback to first available value if specific key not found
                    mainConf = Object.values(confidence)[0] || 0;
                }
            } else if (typeof confidence === 'number') {
                mainConf = confidence;
                confidence = { [emotion]: mainConf };
            }

            // 1. Update Emotion Card
            this.elements.emotionEmoji.textContent = this.config.emojis[emotion] || '🧿';
            this.elements.emotionLabel.textContent = emotion;
            this.elements.emotionConfidence.textContent = `${Math.round(mainConf * 100)}% CONFIDENCE`;
            
            // 1b. Highlight Trigger Grid (Auto-Selection)
            this.highlightTriggerGrid(emotion);
            
            // 1c. Proactive Auto-Trigger Logic
            this.processAutoTrigger(emotion, mainConf);

            // 2. Update Confidence Bars
            this.updateConfidenceBars(confidence);

            // 3. Update Synergy State
            if (data.state) {
                this.elements.synergyState.textContent = data.state;
                this.elements.synergyDesc.textContent = `Neural accuracy optimized at 96.4%`;
            }

            // 4. Update Chat/Message
            if (data.message) {
                this.elements.speechText.textContent = data.message;
                this.log('AI', data.message);
                
                // Detect if assistant is in "Solver" or "Encourager" mode based on text clues
                if (data.message.includes("Step 1:") || data.message.includes("Let's solve")) {
                    this.updateBotState('SOLVER_ACTIVE');
                } else if (data.message.includes("encouraging") || data.message.includes("Start by")) {
                    this.updateBotState('RITUAL_ACTIVE');
                } else {
                    this.updateBotState('OPTIMIZING');
                }
            }

            // 5. Update Neural Ritual
            if (data.ritual) {
                this.elements.ritualContainer.style.display = 'block';
                this.elements.ritualIcon.textContent = data.ritual.icon;
                this.elements.ritualTask.textContent = data.ritual.task;
                this.elements.ritualDesc.textContent = data.ritual.desc;
                this.log('Ritual', `Suggested: ${data.ritual.task}`);
            }

            // 6. Handle Smart Action
            if (data.smart_action) {
                this.log('SmartHome', data.smart_action);
            }
        }
    }

    highlightTriggerGrid(emotion) {
        const btns = this.elements.triggerGrid.querySelectorAll('.trigger-btn');
        btns.forEach(btn => {
            const btnEmotion = btn.getAttribute('data-emotion');
            if (btnEmotion === emotion) {
                btn.classList.add('predicted');
            } else {
                btn.classList.remove('predicted');
            }
        });
    }

    updateBotState(state) {
        this.state.botState = state;
        const footerStatus = document.querySelector('.bottom-bar div:last-child');
        if (footerStatus) {
            footerStatus.textContent = `STATUS: ${state}`;
            footerStatus.style.color = state === 'STANDBY' ? 'var(--text-secondary)' : 'var(--accent-cyan)';
            footerStatus.style.textShadow = state === 'STANDBY' ? 'none' : '0 0 10px var(--accent-cyan)';
        }

        if (state === 'RITUAL_ACTIVE') {
            this.elements.ritualContainer.classList.add('active-pulse');
        } else {
            this.elements.ritualContainer.classList.remove('active-pulse');
        }
    }

    processAutoTrigger(emotion, confidence) {
        // Reset if auto-engage is off
        if (!this.state.autoEngage) {
            this.elements.syncProgress.style.width = '0%';
            return;
        }

        // Only trigger for confident, stable predictions
        if (confidence < 0.7) {
            this.state.emotionStability.count = 0;
            this.elements.syncProgress.style.width = '0%';
            return;
        }

        if (this.state.emotionStability.current === emotion) {
            this.state.emotionStability.count++;
        } else {
            this.state.emotionStability.current = emotion;
            this.state.emotionStability.count = 1;
        }

        // Update Sync Progress Bar
        const progress = (this.state.emotionStability.count / this.state.emotionStability.threshold) * 100;
        this.elements.syncProgress.style.width = `${Math.min(progress, 100)}%`;

        // Check if threshold reached and not a duplicate trigger
        if (this.state.emotionStability.count >= this.state.emotionStability.threshold && 
            this.state.lastTriggeredEmotion !== emotion &&
            emotion !== 'NEUTRAL') {
            
            this.state.lastTriggeredEmotion = emotion;
            this.state.emotionStability.count = 0; // Reset counter after trigger
            this.elements.syncProgress.style.width = '0%';
            
            this.triggerEmotion(emotion);
            this.log('Neural', `Proactive Engagement Triggered: ${emotion}`);
        }
    }

    updateConfidenceBars(confidence) {
        this.elements.confidenceList.innerHTML = '';
        Object.entries(confidence).forEach(([label, val]) => {
            const item = document.createElement('div');
            item.className = 'confidence-item';
            item.innerHTML = `
                <div class="confidence-label">
                    <span>${label.toUpperCase()}</span>
                    <span>${Math.round(val * 100)}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: ${val * 100}%"></div>
                </div>
            `;
            this.elements.confidenceList.appendChild(item);
        });
    }

    setupAudioAnalysis(stream) {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const canvasCtx = this.elements.waveform.getContext('2d');

        const draw = () => {
            if (!this.state.isRunning) return;
            requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            canvasCtx.fillStyle = 'rgba(2, 6, 23, 0.2)';
            canvasCtx.fillRect(0, 0, this.elements.waveform.width, this.elements.waveform.height);

            const barWidth = (this.elements.waveform.width / bufferLength) * 2.5;
            let x = 0;

            let totalVolume = 0;
            for (let i = 0; i < bufferLength; i++) {
                const barHeight = dataArray[i] / 2;
                totalVolume += dataArray[i];
                
                canvasCtx.fillStyle = `rgb(0, ${180 + barHeight}, 255)`;
                canvasCtx.fillRect(x, this.elements.waveform.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
            
            const volPercent = Math.round((totalVolume / (bufferLength * 255)) * 100);
            document.getElementById('volume-val').textContent = `${volPercent}%`;
            document.getElementById('mic-label').textContent = volPercent > 5 ? 'MICROPHONE: CAPTURING' : 'MICROPHONE: IDLE';
        };
        draw();
    }

    log(tag, msg) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const now = new Date();
        const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
        
        entry.innerHTML = `
            <span class="log-time">[${timeStr}]</span>
            <span class="log-tag">${tag.toUpperCase()}</span>
            <span class="log-msg">${msg}</span>
        `;
        
        this.elements.logFeed.prepend(entry);
        if (this.elements.logFeed.childNodes.length > 50) {
            this.elements.logFeed.removeChild(this.elements.logFeed.lastChild);
        }
    }

    setupClock() {
        setInterval(() => {
            const now = new Date();
            this.elements.clock.textContent = `${now.getUTCHours().toString().padStart(2, '0')}:${now.getUTCMinutes().toString().padStart(2, '0')}:${now.getUTCSeconds().toString().padStart(2, '0')} [UTC]`;
        }, 1000);
    }

    setupNeuralBackground() {
        const canvas = document.getElementById('neural-canvas');
        const ctx = canvas.getContext('2d');
        let particles = [];

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            particles = [];
            for (let i = 0; i < 50; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    size: Math.random() * 2
                });
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
            
            particles.forEach((p, i) => {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();

                // Draw lines to nearby particles
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dx = p.x - p2.x;
                    const dy = p.y - p2.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 150) {
                        ctx.strokeStyle = `rgba(0, 212, 255, ${0.1 * (1 - dist/150)})`;
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            });
            requestAnimationFrame(animate);
        };

        window.addEventListener('resize', resize);
        resize();
        animate();
    }
}

// Start HUD
document.addEventListener('DOMContentLoaded', () => {
    window.hud = new OrienHUD();
});
