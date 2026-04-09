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
            },
            behavior_interval: 100, // 10Hz sampling
            translations: {
                'en-US': {
                    VISION: 'VISION',
                    AUDIO_CAPTURE: 'AUDIO_CAPTURE',
                    UPLOAD_RESOURCE: 'UPLOAD_RESOURCE',
                    NEURAL_DECODER: 'NEURAL_DECODER',
                    TEMPORAL_HISTORY: 'TEMPORAL_HISTORY',
                    PREDICTION: 'GENERATE PREDICTION'
                },
                'ta-IN': {
                    VISION: 'பார்வை (Vision)',
                    AUDIO_CAPTURE: 'ஒலி பிடிப்பு (Audio)',
                    UPLOAD_RESOURCE: 'கோப்புப் பதிவேற்றம் (Upload)',
                    NEURAL_DECODER: 'நரம்பியல் குறியீடு (Decoder)',
                    TEMPORAL_HISTORY: 'நேர வரலாறு (History)',
                    PREDICTION: 'கணிப்பைப் பெறுக (Predict)'
                }
            }
        };

        this.behavior = {
            clicks: 0,
            moves: 0,
            positions: [],
            lastSample: Date.now()
        };

        this.elements = {
            video: document.getElementById('webcam-source'),
            visionCanvas: document.getElementById('vision-canvas'),
            captureCanvas: document.getElementById('capture-canvas'),
            orb: document.getElementById('orb-canvas'),
            micBtn: document.getElementById('voice-record-btn'),
            waves: document.getElementById('voice-waves'),
            chat: document.getElementById('chat-feed'),
            chatInput: document.getElementById('chat-input'),
            chatSend: document.getElementById('chat-send'),
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
            language: localStorage.getItem('orien_lang') || 'en-US',
            hasPendingData: false,
            pendingBlob: null,
            pendingType: null,
            history: JSON.parse(localStorage.getItem('orien_history') || '[]'),
            ws: null,
            isSpeaking: false,
            isListening: false
        };

        this.init();
    }

    async init() {
        this.setupThreeJS();
        this.setupAudioVisualization();
        this.setupChart();
        this.setupWebSocket();
        this.bindEvents();
        this.startClock();
        this.updateLangUI();
        this.setUIState('IDLE');
        
        // [AUTO-INIT] Request permissions after brief stabilization delay
        setTimeout(() => this.requestPermissions(), 1500);
    }

    async requestPermissions() {
        console.log("💎 ORIEN | Calibrating Neural Sensors (Camera/Mic)...");
        try {
            // Sequential request for clarity
            await this.toggleWebcam(); // Triggers Camera + Vision Loop
        } catch (e) {
            console.warn("Camera auto-init bypassed or blocked.");
        }
    }

    /** 🛰️ NEURAL BRIDGE (WebSocket) */
    setupWebSocket() {
        this.state.ws = new WebSocket(this.config.ws_url);
        
        this.state.ws.onopen = () => {
            console.log("💎 ORIEN | Neural Bridge ACTIVE.");
            this.addChatMessage('System', "Neural Bridge Established.");
        };

        this.state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.status === 'ok' && data.message) {
                    if (data.state) this.updateAppState(data.state);
                    
                    if (data.message === "Service temporarily unavailable. Retrying...") {
                        this.addChatMessage('System', "Neural Routing Failure: Attempting auto-rotation...");
                    } else {
                        this.addChatMessage('AI', data.message);
                    }
                    
                    if (data.insight) this.showInsight(data.insight);
                    if (data.entropy !== undefined) this.updateEntropy(data.entropy);
                }
            } catch (e) {
                console.error("Neural Bridge Sync Error:", e);
            }
        };

        this.state.ws.onclose = () => {
            console.warn("⚠️ ORIEN | Neural Bridge DISCONNECTED. Retrying...");
            setTimeout(() => this.setupWebSocket(), 5000);
        };
    }

    /** 🌐 LANGUAGE BONDING */
    toggleLanguage() {
        this.state.language = this.state.language === 'en-US' ? 'ta-IN' : 'en-US';
        localStorage.setItem('orien_lang', this.state.language);
        this.updateLangUI();
        this.addChatMessage('System', `Language calibrated: ${this.state.language === 'en-US' ? 'English' : 'Tamil (தமிழ்)'}`);
    }

    updateLangUI() {
        const display = document.getElementById('lang-display');
        if (display) display.textContent = this.state.language.toUpperCase();

        // [SOTA] High-Fidelity Multilingual Label Sync
        const labels = this.config.translations[this.state.language] || this.config.translations['en-US'];
        
        const map = {
            'VISION': '.control-node span',
            'AUDIO_CAPTURE': '.panel-header',
            'UPLOAD_RESOURCE': '.panel-header',
            'NEURAL_DECODER': '.panel-header',
            'TEMPORAL_HISTORY': '.panel-header'
        };

        // Header Labels
        const headers = document.querySelectorAll('.panel-header');
        if (headers.length >= 4) {
            headers[0].textContent = labels.VISION || 'VISION_CAPTURE'; // Actually Vision is a child usually
            headers[1].textContent = labels.AUDIO_CAPTURE;
            headers[2].textContent = labels.UPLOAD_RESOURCE;
            headers[3].textContent = labels.NEURAL_DECODER;
            headers[4].textContent = labels.TEMPORAL_HISTORY;
        }

        const predBtn = document.getElementById('generate-prediction');
        if (predBtn) predBtn.textContent = labels.PREDICTION;
    }

    updateAppState(state) {
        const s = state.toUpperCase();
        const prevClass = document.body.className;
        document.body.className = `state-${s.toLowerCase()}`;
        this.elements.indicators.state.textContent = `SYNC_${s}`;
        
        // 🔮 GLITCH EFFECT ON TRANSITION
        if (prevClass !== document.body.className) {
            document.body.style.filter = "invert(1) hue-rotate(90deg) contrast(2)";
            setTimeout(() => { document.body.style.filter = "none"; }, 80);
        }

        // Aura resonance based on state
        const aura = document.getElementById('neural-aura');
        if (aura) aura.className = s === 'HAPPY' || s === 'STRESSED' ? 'active' : '';
        
        // Update Orb color based on state
        if (this.orbMaterial) {
            const colors = {
                HAPPY: 0xfbbf24, SAD: 0x94a3b8, ANGRY: 0xef4444, STRESSED: 0xf97316, NEUTRAL: 0x00d4ff
            };
            this.orbMaterial.color.setHex(colors[s] || 0x00d4ff);
            this.orbMaterial.emissive.setHex(colors[s] || 0x00d4ff);
        }
    }

    showInsight(text) {
        const overlay = document.getElementById('insight-overlay');
        const content = document.getElementById('insight-text');
        if (!overlay || !content) return;

        content.textContent = text;
        overlay.classList.add('active');
        
        // Reset after 8 seconds
        clearTimeout(this.insightTimer);
        this.insightTimer = setTimeout(() => {
            overlay.classList.remove('active');
        }, 8000);
    }

    updateEntropy(entropy) {
        const bar = document.getElementById('entropy-bar');
        const val = document.getElementById('entropy-value');
        if (!bar || !val) return;

        // Entropy typically ranges from 0 (stable) to ~2.3 (unstable for 5 states)
        const percentage = Math.min(100, (entropy / 2.32) * 100);
        bar.style.width = `${percentage}%`;
        bar.style.background = entropy > 1.2 ? 'var(--entropy-high)' : 'var(--entropy-low)';
        bar.style.boxShadow = `0 0 10px ${entropy > 1.2 ? 'var(--entropy-high)' : 'var(--entropy-low)'}`;
        val.textContent = entropy.toFixed(2);
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
        const base64 = this.elements.captureCanvas.toDataURL('image/jpeg', 0.5);
        
        this.state.pendingBlob = base64;
        this.state.pendingType = 'FACE';
        this.state.hasPendingData = true;
        this.updatePredictBtnState();

        // 🛰️ REAL-TIME STREAMING TO BRIDGE (Vision + Behavior)
        this.streamFrameToBridge(base64);
    }

    streamFrameToBridge(base64) {
        if (this.state.ws && this.state.ws.readyState === WebSocket.OPEN) {
            this.state.ws.send(JSON.stringify({
                type: 'FRAME',
                image: base64,
                language: this.state.language,
                focus_level: 1.0, 
                behavior: this.sampleBehavior()
            }));
        }
    }

    sampleBehavior() {
        const now = Date.now();
        const dt = (now - this.behavior.lastSample) / 1000;
        
        const speed = dt > 0 ? this.behavior.moves / dt : 0;
        const jitter = this.calculateJitter();
        
        const metrics = {
            mouse_speed: speed,
            click_count: this.behavior.clicks,
            jitter: jitter,
            wpm: 0, 
            backspaces: 0,
            window_switches: 0
        };

        // Reset counters for next sample
        this.behavior.clicks = 0;
        this.behavior.moves = 0;
        this.behavior.positions = [];
        this.behavior.lastSample = now;
        
        return metrics;
    }

    calculateJitter() {
        if (this.behavior.positions.length < 2) return 0;
        let sum = 0;
        for (let i = 1; i < this.behavior.positions.length; i++) {
            const dx = this.behavior.positions[i].x - this.behavior.positions[i-1].x;
            const dy = this.behavior.positions[i].y - this.behavior.positions[i-1].y;
            sum += Math.sqrt(dx*dx + dy*dy);
        }
        return sum / this.behavior.positions.length;
    }

    /** 🎙️ SPEECH TO TEXT (STT) */
    async toggleRecording() {
        if (this.state.isListening) {
            this.stopSTT();
        } else {
            this.startSTT();
        }
    }

    startSTT() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            this.showError("Speech Recognition not supported in this browser.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.lang = this.state.language;
        this.recognition.continuous = false;
        this.recognition.interimResults = true;

        this.recognition.onstart = () => {
            this.state.isListening = true;
            this.setUIState('RECORDING');
            this.elements.micBtn.classList.add('recording');
            this.elements.indicators.record.textContent = 'LISTENING...';
        };

        this.recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
            
            if (event.results[0].isFinal) {
                this.elements.chatInput.value = transcript;
                this.submitChat();
                this.stopSTT();
            }
        };

        this.recognition.onerror = (e) => {
            console.error("STT Error:", e);
            this.stopSTT();
        };

        this.recognition.onend = () => {
            this.stopSTT();
        };

        this.recognition.start();
        
        // Start visualizer
        this.startAudioVisualizer();
    }

    stopSTT() {
        if (this.recognition) {
            this.recognition.stop();
            this.recognition = null;
        }
        this.state.isListening = false;
        this.setUIState('IDLE');
        this.elements.micBtn.classList.remove('recording');
        this.elements.indicators.record.textContent = 'IDLE';
    }

    async startAudioVisualizer() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.setupAudioAnalysis(stream);
        } catch (e) {
            console.warn("Visualizer failed:", e);
        }
    }

    /** 🔊 TEXT TO SPEECH (TTS) */
    speak(text) {
        if (!window.speechSynthesis) return;
        
        // Stop any current speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = this.state.language;
        
        // Adjust tone based on emotion state
        const state = document.body.className.split('-')[1] || 'neutral';
        switch(state) {
            case 'sad': utterance.pitch = 0.8; utterance.rate = 0.8; break;
            case 'happy': utterance.pitch = 1.2; utterance.rate = 1.1; break;
            case 'angry': utterance.pitch = 0.7; utterance.rate = 1.0; break;
            case 'stressed': utterance.pitch = 1.0; utterance.rate = 1.2; break;
            default: utterance.pitch = 1.0; utterance.rate = 1.0;
        }

        utterance.onstart = () => { this.state.isSpeaking = true; };
        utterance.onend = () => { this.state.isSpeaking = false; };
        
        window.speechSynthesis.speak(utterance);
    }

    render() {
        // ... (render logic)
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
            formData.append('language', this.state.language);
            
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
            console.error(e);
            this.addChatMessage('AI', "I'm having trouble connecting to my neural core right now. One moment while I recalibrate.");
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
        
        // Chat Input
        this.elements.chatSend.addEventListener('click', () => this.submitChat());
        this.elements.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.submitChat();
        });

        const langToggle = document.getElementById('lang-toggle');
        if (langToggle) langToggle.addEventListener('click', () => this.toggleLanguage());

        this.elements.dropZone.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));
        
        // Behavioral Tracking
        window.addEventListener('mousemove', (e) => {
            this.behavior.moves++;
            this.behavior.positions.push({ x: e.clientX, y: e.clientY });
            if (this.behavior.positions.length > 50) this.behavior.positions.shift();
        });
        window.addEventListener('click', () => this.behavior.clicks++);

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

    submitChat() {
        const text = this.elements.chatInput.value.trim();
        if (!text) return;
        
        this.addChatMessage('User', text);
        this.elements.chatInput.value = '';
        
        if (this.state.ws && this.state.ws.readyState === WebSocket.OPEN) {
            this.state.ws.send(JSON.stringify({
                query: text,
                language: this.state.language,
                face_emotion: this.lastDetectedEmotion || 'Neutral',
                behavior: this.sampleBehavior()
            }));

            // Show Thinking Indicator immediately
            const feed = this.elements.chat;
            const b = document.createElement('div');
            b.className = 'bubble ai typing-bubble current-thinking';
            b.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
            feed.appendChild(b);
            feed.scrollTop = feed.scrollHeight;
        }
    }

    async addChatMessage(role, text) {
        const feed = this.elements.chat;
        
        // Remove thinking indicator if AI response arrives
        if (role === 'AI') {
            const thinking = feed.querySelector('.current-thinking');
            if (thinking) thinking.remove();
        }

        const b = document.createElement('div');
        b.className = `bubble ${role.toLowerCase()}`;
        
        if (role === 'AI') {
            b.classList.add('typing-bubble');
            const typingIndicator = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
            b.innerHTML = typingIndicator;
            feed.appendChild(b);
            feed.scrollTop = feed.scrollHeight;
            
            // Artificial delay before typing starts
            await new Promise(r => setTimeout(r, 600 + Math.random() * 600));
            
            b.innerHTML = '';
            b.classList.remove('typing-bubble');
            
            // Character-by-character typing
            let currentText = '';
            for (const char of text) {
                currentText += char;
                b.textContent = currentText;
                feed.scrollTop = feed.scrollHeight;
                // Variable delay for human-like rhythm
                await new Promise(r => setTimeout(r, 15 + Math.random() * 25));
            }
            
            this.speak(text);
        } else {
            b.textContent = text;
            feed.appendChild(b);
        }
        feed.scrollTop = feed.scrollHeight;
    }

    /** 🔊 TEXT TO SPEECH (TTS) ENHANCED */
    speak(text) {
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        
        let selectedVoice = null;
        if (this.state.language.startsWith('ta')) {
            selectedVoice = voices.find(v => v.lang.startsWith('ta')) || voices.find(v => v.lang.includes('India'));
        } else {
            selectedVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Premium'))) || voices[0];
        }
        
        if (selectedVoice) utterance.voice = selectedVoice;
        utterance.lang = this.state.language;
        
        const state = document.body.className.split('-')[1] || 'neutral';
        const profiles = {
            sad: { pitch: 0.85, rate: 0.85 },
            happy: { pitch: 1.15, rate: 1.1 },
            angry: { pitch: 0.75, rate: 1.0 },
            stressed: { pitch: 1.0, rate: 1.25 },
            neutral: { pitch: 1.0, rate: 1.0 }
        };
        
        const p = profiles[state] || profiles.neutral;
        utterance.pitch = p.pitch;
        utterance.rate = p.rate;
        window.speechSynthesis.speak(utterance);
    }

    setupThreeJS() {
        const sc = new THREE.Scene(), cm = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        const rd = new THREE.WebGLRenderer({ canvas: this.elements.orb, alpha: true, antialias: true });
        rd.setSize(400, 400); cm.position.z = 2.5;
        
        // 🔮 NERVE ORB
        this.orbMaterial = new THREE.MeshPhongMaterial({ color: '#00d4ff', wireframe: true, transparent: true, opacity: 0.3, emissive: '#00d4ff', emissiveIntensity: 0.4 });
        const orb = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 4), this.orbMaterial);
        sc.add(orb);
        
        // 🌌 NEURAL GRID
        const grid = new THREE.GridHelper(10, 20, 0x00d4ff, 0x011627);
        grid.rotation.x = Math.PI / 2;
        grid.position.z = -2;
        grid.material.transparent = true;
        grid.material.opacity = 0.1;
        sc.add(grid);

        sc.add(new THREE.AmbientLight(0x404040));
        const lt = new THREE.PointLight(0xffffff, 1.5, 10); lt.position.set(2, 2, 5); sc.add(lt);
        
        const anim = () => { 
            requestAnimationFrame(anim); 
            orb.rotation.y += 0.005; 
            orb.rotation.x += 0.002;
            grid.position.z += 0.001; if(grid.position.z > 0) grid.position.z = -2;
            rd.render(sc, cm); 
        };
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
