/**
 * 💎 ORIEN | FRONTEND CONTROL SYSTEM [MULTIMODAL SYNC]
 * Capture → Send → Await → Sanitize → Render
 */

class NeuralControlHub {
    constructor() {
        this.config = {
            ws_url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8000/ws/neural`,
            api_base: `${window.location.protocol}//${window.location.hostname}:8000/api`,
            vision_fps: 1, // Protocol: 1-2 fps throttled
            emojis: {
                HAPPY: '😊', SAD: '😢', ANGRY: '😠', NEUTRAL: '😐', STRESSED: '😰', SURPRISE: '😲', DISGUST: '🤢'
            }
        };

        this.elements = {
            video: document.getElementById('webcam-source'),
            visionCanvas: document.getElementById('vision-canvas'),
            captureCanvas: document.getElementById('capture-canvas'),
            orb: document.getElementById('orb-canvas'),
            micBtn: document.getElementById('voice-record-btn'),
            waves: document.getElementById('voice-waves'),
            chat: document.getElementById('chat-feed'),
            fileInput: document.getElementById('file-input'),
            dropZone: document.getElementById('drop-zone'),
            predictBtn: document.getElementById('generate-prediction'),
            webcamToggle: document.getElementById('webcam-toggle'),
            results: {
                panel: document.getElementById('result-card'),
                emoji: document.getElementById('res-emoji'),
                emotion: document.getElementById('res-emotion'),
                mode: document.getElementById('res-mode'),
                bars: document.getElementById('confidence-bars')
            },
            indicators: {
                state: document.getElementById('app-state-indicator'),
                record: document.getElementById('record-status'),
                overlay: document.getElementById('recording-overlay')
            }
        };

        this.state = {
            ui: 'IDLE', // IDLE | RECORDING | PROCESSING | DISPLAY_RESULT | ERROR
            visionActive: false,
            mediaRecorder: null,
            audioChunks: [],
            lastFrame: 0,
            language: 'en-US',
            hasPendingData: false,
            pendingBlob: null,
            pendingType: null,
            history: JSON.parse(localStorage.getItem('orien_history') || '[]')
        };

        this.init();
    }

    async init() {
        this.setupThreeJS();
        this.setupAudioVisualization();
        this.setupChart();
        this.bindEvents();
        this.startClock();
        this.setUIState('IDLE');
    }

    /** 🎛️ UI STATE CONTROLLER */
    setUIState(newState) {
        this.state.ui = newState;
        this.elements.indicators.state.textContent = `SYNC_${newState}`;
        
        // Locking/Unlocking UI
        if (newState === 'PROCESSING') {
            document.body.classList.add('lock-ui');
            this.elements.predictBtn.classList.add('disabled');
            this.elements.predictBtn.disabled = true;
            this.elements.predictBtn.textContent = 'ANALYZING...';
        } else {
            document.body.classList.remove('lock-ui');
            this.elements.predictBtn.textContent = 'GENERATE PREDICTION';
            this.updatePredictBtnState();
        }
    }

    updatePredictBtnState() {
        if (this.state.hasPendingData) {
            this.elements.predictBtn.classList.remove('disabled');
            this.elements.predictBtn.disabled = false;
        } else {
            this.elements.predictBtn.classList.add('disabled');
            this.elements.predictBtn.disabled = true;
        }
    }

    /** 🛡️ DATA SANITIZATION LAYER (STRICT) */
    sanitize(data) {
        if (!data || typeof data !== 'object') return null;
        
        const forbidden = ["model", "version", "pipeline", "debug", "meta", "latency", "device"];
        const keys = Object.keys(data);
        
        const hasLeak = keys.some(k => forbidden.includes(k.toLowerCase())) || 
                        JSON.stringify(data).toLowerCase().includes('model');

        if (hasLeak) {
            console.warn("🛡️ SANITIZATION: Internal metadata leak blocked.");
            // Strip forbidden fields
            forbidden.forEach(f => delete data[f]);
            
            // If it's a critical leak or malformed, fallback
            if (!data.emotion) return { status: "ok", emotion: "NEUTRAL", confidence: { neutral: 1.0 }, message: "I'm here to help you." };
        }
        return data;
    }

    /** 📸 VISION CAPTURE */
    async toggleWebcam() {
        if (this.state.visionActive) {
            const stream = this.elements.video.srcObject;
            if (stream) stream.getTracks().forEach(track => track.stop());
            this.elements.video.srcObject = null;
            this.state.visionActive = false;
            this.elements.webcamToggle.classList.remove('active');
            this.elements.indicators.overlay.classList.add('hidden');
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                this.elements.video.srcObject = stream;
                this.state.visionActive = true;
                this.elements.webcamToggle.classList.add('active');
                this.elements.indicators.overlay.classList.remove('hidden');
                this.startVisionLoop();
            } catch (e) {
                this.showError("Permission required to continue (Camera)");
            }
        }
    }

    startVisionLoop() {
        const loop = () => {
            if (!this.state.visionActive) return;
            const ctx = this.elements.visionCanvas.getContext('2d');
            this.elements.visionCanvas.width = this.elements.visionCanvas.clientWidth;
            this.elements.visionCanvas.height = this.elements.visionCanvas.clientHeight;
            
            ctx.drawImage(this.elements.video, 0, 0, this.elements.visionCanvas.width, this.elements.visionCanvas.height);
            
            const now = performance.now();
            if (now - this.state.lastFrame > (1000 / this.config.vision_fps)) {
                this.state.lastFrame = now;
                this.captureFrameForPrediction();
            }
            requestAnimationFrame(loop);
        };
        loop();
    }

    captureFrameForPrediction() {
        const ctx = this.elements.captureCanvas.getContext('2d');
        this.elements.captureCanvas.width = 120;
        this.elements.captureCanvas.height = 120;
        ctx.drawImage(this.elements.video, 0, 0, 120, 120);
        this.state.pendingBlob = this.elements.captureCanvas.toDataURL('image/jpeg', 0.5);
        this.state.pendingType = 'FACE';
        this.state.hasPendingData = true;
        this.updatePredictBtnState();
    }

    /** 🎙️ AUDIO CAPTURE */
    async toggleRecording() {
        if (this.state.ui === 'RECORDING') {
            this.state.mediaRecorder.stop();
            this.setUIState('IDLE');
            this.elements.micBtn.classList.remove('recording');
            this.elements.indicators.record.textContent = 'IDLE';
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.state.mediaRecorder = new MediaRecorder(stream);
                this.state.audioChunks = [];
                
                this.state.mediaRecorder.ondataavailable = (e) => this.state.audioChunks.push(e.data);
                this.state.mediaRecorder.onstop = () => {
                    this.state.pendingBlob = new Blob(this.state.audioChunks, { type: 'audio/wav' });
                    this.state.pendingType = 'VOICE';
                    this.state.hasPendingData = true;
                    this.updatePredictBtnState();
                    stream.getTracks().forEach(t => t.stop());
                };

                this.state.mediaRecorder.start();
                this.setUIState('RECORDING');
                this.elements.micBtn.classList.add('recording');
                this.elements.indicators.record.textContent = 'RECORDING';
                this.setupAudioAnalysis(stream);
            } catch (e) {
                this.showError("Permission required to continue (Microphone)");
            }
        }
    }

    /** 📁 FILE CAPTURE */
    handleFileUpload(file) {
        if (!file) return;
        const validTypes = ['audio/wav', 'audio/mpeg', 'image/jpeg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            this.showError("Invalid file type. Use .wav, .mp3, .jpg, or .png");
            return;
        }

        this.state.pendingBlob = file;
        this.state.pendingType = file.type.startsWith('audio') ? 'VOICE' : 'FACE';
        this.state.hasPendingData = true;
        this.updatePredictBtnState();
        this.addChatMessage('System', `File loaded: ${file.name}`);
        
        // Protocol: Auto-trigger prediction on file upload
        this.executePrediction();
    }

    /** 🧠 PREDICTION EXECUTION */
    async executePrediction() {
        if (!this.state.hasPendingData) return;
        
        this.setUIState('PROCESSING');
        
        try {
            const formData = new FormData();
            if (typeof this.state.pendingBlob === 'string') {
                formData.append('image', this.state.pendingBlob);
            } else {
                formData.append('file', this.state.pendingBlob);
            }
            formData.append('mode', this.state.pendingType);
            
            const endpoint = this.state.pendingType === 'VOICE' ? '/predict/voice' : '/predict/face';
            const response = await fetch(`${this.config.api_base}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("API_FAIL");
            
            let data = await response.json();
            data = this.sanitize(data);
            
            if (data.status === 'ok') {
                this.renderResults(data);
                this.setUIState('DISPLAY_RESULT');
            } else {
                throw new Error("SERVER_FAIL");
            }
        } catch (e) {
            this.showError("Something went wrong. Try again.");
            this.setUIState('ERROR');
        } finally {
            this.state.hasPendingData = false;
        }
    }

    /** 📊 RENDERING LABS */
    renderResults(data) {
        const { emotion, confidence, mode } = data;
        
        this.elements.results.panel.classList.remove('hidden');
        this.elements.results.emoji.textContent = this.config.emojis[emotion.toUpperCase()] || '🧿';
        this.elements.results.emotion.textContent = emotion.toUpperCase();
        this.elements.results.mode.textContent = `MODE: ${mode}`;
        
        // Confidence Bars
        this.elements.results.bars.innerHTML = '';
        Object.entries(confidence).forEach(([emo, val]) => {
            const row = document.createElement('div');
            row.className = 'confidence-row';
            row.style.margin = '10px 0';
            row.innerHTML = `
                <div style="display:flex; justify-content:space-between; font-size:10px; margin-bottom:4px;">
                    <span>${emo.toUpperCase()}</span>
                    <span>${(val * 100).toFixed(2)}%</span>
                </div>
                <div class="confidence-bar" style="height:4px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden;">
                    <div style="width:${val * 100}%; height:100%; background:var(--primary); transition:width 1s ease;"></div>
                </div>
            `;
            this.elements.results.bars.appendChild(row);
        });

        // History Hub
        this.state.history.push({ emotion, timestamp: Date.now() });
        if (this.state.history.length > 20) this.state.history.shift();
        localStorage.setItem('orien_history', JSON.stringify(this.state.history));
        this.updateChart();
    }

    updateChart() {
        if (!this.chart) return;
        const labels = this.state.history.map(h => new Date(h.timestamp).toLocaleTimeString());
        const data = this.state.history.map(h => {
             const map = { HAPPY: 5, SURPRISE: 4, NEUTRAL: 3, SAD: 2, ANGRY: 1, STRESSED: 0 };
             return map[h.emotion.toUpperCase()] ?? 3;
        });

        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = data;
        this.chart.update();
    }

    setupChart() {
        const ctx = document.getElementById('history-chart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Emotion Synergy',
                    data: [],
                    borderColor: '#00d4ff',
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 5, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { display: false } },
                    x: { ticks: { display: false }, grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    /** 🎨 INFRASTRUCTURE */
    bindEvents() {
        this.elements.webcamToggle.addEventListener('click', () => this.toggleWebcam());
        this.elements.micBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.predictBtn.addEventListener('click', () => this.executePrediction());
        
        this.elements.dropZone.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));
        
        // Drag n Drop
        this.elements.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); this.elements.dropZone.classList.add('active'); });
        this.elements.dropZone.addEventListener('dragleave', () => this.elements.dropZone.classList.remove('active'));
        this.elements.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.dropZone.classList.remove('active');
            this.handleFileUpload(e.dataTransfer.files[0]);
        });
    }

    showError(msg) {
        this.addChatMessage('System', `⚠️ ${msg}`);
        // Protocol: Fallback for detected leakage or failure
        this.elements.results.emotion.textContent = "IDLE_STATE";
    }

    addChatMessage(role, text) {
        const b = document.createElement('div');
        b.className = `bubble ${role.toLowerCase()}`;
        b.textContent = text;
        this.elements.chat.appendChild(b);
        this.elements.chat.scrollTop = this.elements.chat.scrollHeight;
    }

    setupThreeJS() {
        const sc = new THREE.Scene(), cm = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        const rd = new THREE.WebGLRenderer({ canvas: this.elements.orb, alpha: true, antialias: true });
        rd.setSize(400, 400); cm.position.z = 2.5;
        const mat = new THREE.MeshPhongMaterial({ color: '#00d4ff', wireframe: true, transparent: true, opacity: 0.3, emissive: '#00d4ff', emissiveIntensity: 0.4 });
        const orb = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 4), mat);
        sc.add(orb); sc.add(new THREE.AmbientLight(0x404040));
        const lt = new THREE.PointLight(0xffffff, 1.5, 10); lt.position.set(2, 2, 5); sc.add(lt);
        const anim = () => { requestAnimationFrame(anim); orb.rotation.y += 0.005; rd.render(sc, cm); };
        anim();
    }

    setupAudioVisualization() {
        for (let i = 0; i < 20; i++) {
            const b = document.createElement('div');
            b.className = 'bar';
            this.elements.waves.appendChild(b);
        }
    }

    setupAudioAnalysis(s) {
        const ctx = new AudioContext(), src = ctx.createMediaStreamSource(s), ans = ctx.createAnalyser();
        src.connect(ans); const dat = new Uint8Array(ans.frequencyBinCount);
        const render = () => {
             ans.getByteFrequencyData(dat);
             const bars = this.elements.waves.children;
             for (let i = 0; i < bars.length; i++) {
                 const v = dat[i % dat.length] / 255;
                 bars[i].style.height = `${4 + v * 30}px`;
             }
             if (this.state.ui === 'RECORDING') requestAnimationFrame(render);
        };
        render();
    }

    startClock() {
        setInterval(() => {
            document.getElementById('nav-clock').textContent = new Date().toLocaleTimeString() + ' [UTC]';
        }, 1000);
    }
}

window.onload = () => new NeuralControlHub();
