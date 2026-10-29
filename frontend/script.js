let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("status");
const resultText = document.getElementById("result");

recordBtn.onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  audioChunks = [];

  processor.onaudioprocess = e => {
    audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  statusText.innerText = "Recording...";
  recordBtn.disabled = true;
  stopBtn.disabled = false;

  stopBtn.onclick = async () => {
    processor.disconnect();
    source.disconnect();

    // Convert Float32 to WAV
    const wavBuffer = encodeWAV(audioChunks, audioContext.sampleRate);
    const blob = new Blob([wavBuffer], { type: "audio/wav" });

    const formData = new FormData();
    formData.append("file", blob, "voice.wav");

    statusText.innerText = "Sending to AI...";

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    resultText.innerText =
      `Stress Level: ${data.stress_level} | Confidence: ${data.confidence}`;

    statusText.innerText = "Done";

    recordBtn.disabled = false;
    stopBtn.disabled = true;
  };
};

function encodeWAV(samples, sampleRate) {
  let bufferLength = samples.length * samples[0].length * 2 + 44;
  let buffer = new ArrayBuffer(bufferLength);
  let view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * samples[0].length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * samples[0].length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    for (let j = 0; j < samples[i].length; j++, offset += 2) {
      let s = Math.max(-1, Math.min(1, samples[i][j]));
      view.setInt16(offset, s * 0x7fff, true);
    }
  }

  return buffer;
}
