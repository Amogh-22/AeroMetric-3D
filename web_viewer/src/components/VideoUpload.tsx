"use client";

import React, { useState, useRef, useEffect } from "react";
import { 
  Upload, 
  Video, 
  CheckCircle, 
  Loader2, 
  AlertCircle, 
  Server, 
  Link as LinkIcon, 
  Sparkles,
  Layers,
  ChevronRight
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface VideoUploadProps {
  onUploadComplete?: (url: string) => void;
  onLoadDirectModel?: (url: string, metadataUrl?: string) => void;
}

export default function VideoUpload({ 
  onUploadComplete,
  onLoadDirectModel 
}: VideoUploadProps) {
  const [activeTab, setActiveTab] = useState<"pipeline" | "direct">("pipeline");
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [apiUrl, setApiUrl] = useState<string>("");
  const [directUrl, setDirectUrl] = useState<string>("");
  const [uploadState, setUploadState] = useState<"idle" | "uploading" | "processing" | "success" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [pipelineStep, setPipelineStep] = useState<string>("Initializing GPU Pipeline...");
  const inputRef = useRef<HTMLInputElement>(null);

  // Load saved API URL from localStorage if available
  useEffect(() => {
    try {
      const saved = localStorage.getItem("aerometric_api_url");
      if (saved) setApiUrl(saved);
    } catch (e) {}
  }, []);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith("video/") || droppedFile.name.endsWith(".mp4") || droppedFile.name.endsWith(".mov")) {
        setFile(droppedFile);
      } else {
        setErrorMessage("Please upload a valid MP4 or MOV video file.");
        setUploadState("error");
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    if (!apiUrl) {
      setErrorMessage("Please enter your active Colab Tunnel URL (Localtunnel or Ngrok).");
      setUploadState("error");
      return;
    }

    let cleanUrl = apiUrl.trim();
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.slice(0, -1);
    }

    try {
      localStorage.setItem("aerometric_api_url", cleanUrl);
    } catch (e) {}

    setUploadState("uploading");
    setPipelineStep("Packaging scripts & streaming footage to GPU...");
    
    const formData = new FormData();
    formData.append("video", file);
    formData.append("ngrokUrl", cleanUrl);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }
      
      const data = await response.json();
      const datasetName = data.datasetName;

      setUploadState("processing");
      setPipelineStep("SfM Feature Matching & Metric Depth Fusion...");

      let pollAttempts = 0;
      const pollModel = async () => {
        try {
          pollAttempts++;
          const statusEndpoint = `${cleanUrl}/status/${datasetName}`;
          
          if (pollAttempts > 3) {
            setPipelineStep("Fusing point cloud & generating Poisson surface mesh...");
          }

          const statusRes = await fetch(statusEndpoint, {
            headers: { 
              "ngrok-skip-browser-warning": "true",
              "Bypass-Tunnel-Reminder": "true"
            }
          });
          
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            
            if (statusData.status === "failed") {
              setUploadState("error");
              let err = statusData.error || statusData.error_details || statusData.log;
              if (!err) {
                try {
                  const metaRes = await fetch(`${cleanUrl}/metadata/${datasetName}`, {
                    headers: { "Bypass-Tunnel-Reminder": "true", "ngrok-skip-browser-warning": "true" }
                  });
                  if (metaRes.ok) {
                    const metaData = await metaRes.json();
                    if (metaData.error) err = metaData.error;
                  }
                } catch (_) {}
              }
              setErrorMessage(err || "Pipeline failed on Colab. Please check the notebook output cells.");
              return;
            }
            
            if (statusData.status === "processing" || statusData.status === "running" || statusData.status === "pending") {
              if (statusData.progress) {
                setPipelineStep(statusData.progress);
              }
              setTimeout(pollModel, 3000);
              return;
            }

            if (statusData.status === "completed") {
              setUploadState("success");
              if (onUploadComplete) {
                onUploadComplete(`${cleanUrl}/model/${datasetName}`);
              }
              return;
            }
          }
          
          setTimeout(pollModel, 4000);
        } catch (err) {
          setTimeout(pollModel, 4000);
        }
      };
      
      pollModel();

    } catch (error: any) {
      setErrorMessage(error.message || "Failed to reach backend server.");
      setUploadState("error");
    }
  };

  const handleDirectSubmit = () => {
    if (!directUrl.trim()) return;
    const url = directUrl.trim();
    if (onLoadDirectModel) {
      onLoadDirectModel(url);
    } else if (onUploadComplete) {
      onUploadComplete(url);
    }
  };

  const handleLoadSample = () => {
    if (onLoadDirectModel) {
      onLoadDirectModel("/sample_model.glb", "/sample_metadata.json");
    } else if (onUploadComplete) {
      onUploadComplete("/sample_model.glb");
    }
  };

  const reset = () => {
    setFile(null);
    setUploadState("idle");
    setErrorMessage("");
  };

  return (
    <div className="w-full bg-[#0a152e]/90 backdrop-blur-2xl border border-blue-500/25 rounded-2xl shadow-[0_8px_32px_rgba(2,6,23,0.7)] p-6 relative overflow-hidden">
      {/* Decorative Glow */}
      <div className="absolute -top-16 -right-16 w-36 h-36 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-16 -left-16 w-36 h-36 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-blue-500/20 mb-5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-cyan-400">
            <Upload className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white tracking-wide">3D Model Input</h2>
            <p className="text-[11px] text-blue-300/70 font-mono">DRONE PHOTOGRAMMETRY</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-950/80 rounded-full border border-blue-500/30 text-[10px] font-semibold text-cyan-300">
          <Server className="w-3 h-3 text-cyan-400" />
          COLAB GPU
        </div>
      </div>

      {/* Mode Tabs */}
      <div className="flex rounded-xl bg-[#060d1f] p-1 border border-blue-500/20 mb-5">
        <button
          onClick={() => setActiveTab("pipeline")}
          className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center justify-center gap-1.5 ${
            activeTab === "pipeline"
              ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
              : "text-blue-300/70 hover:text-white"
          }`}
        >
          <Video className="w-3.5 h-3.5" /> Drone Video
        </button>
        <button
          onClick={() => setActiveTab("direct")}
          className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center justify-center gap-1.5 ${
            activeTab === "direct"
              ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
              : "text-blue-300/70 hover:text-white"
          }`}
        >
          <LinkIcon className="w-3.5 h-3.5" /> Direct / Demo
        </button>
      </div>

      {activeTab === "pipeline" ? (
        <AnimatePresence mode="wait">
          {uploadState === "idle" || uploadState === "error" ? (
            <motion.div
              key="pipeline-upload"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex flex-col gap-4"
            >
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-semibold text-blue-200">
                    Colab Tunnel URL
                  </label>
                  <span className="text-[10px] text-blue-400 font-mono">loca.lt / ngrok</span>
                </div>
                <input
                  type="text"
                  placeholder="https://sharp-walrus-42.loca.lt"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="w-full bg-[#060d1f]/90 border border-blue-500/30 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-blue-400/40 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/30 font-mono transition-all"
                />
              </div>
              
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 ${
                  dragActive 
                    ? "border-cyan-400 bg-cyan-500/10 scale-[1.01]" 
                    : "border-blue-500/30 hover:border-blue-400/60 bg-[#060e22]/50 hover:bg-[#08122a]/70"
                }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleChange}
                  className="hidden"
                />
                <div className="w-10 h-10 rounded-full bg-blue-500/10 border border-blue-400/30 flex items-center justify-center mb-2.5">
                  <Video className={`w-5 h-5 ${dragActive ? "text-cyan-400" : "text-blue-400"}`} />
                </div>
                <p className="text-xs font-semibold text-white">
                  {file ? file.name : "Select or Drop Drone Video"}
                </p>
                <p className="text-[11px] text-blue-300/60 mt-1">
                  {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "MP4 or MOV footage"}
                </p>
              </div>

              {uploadState === "error" && (
                <div className="flex items-start gap-2.5 text-red-300 text-xs p-3 bg-red-950/40 border border-red-500/30 rounded-xl">
                  <AlertCircle className="w-4 h-4 shrink-0 text-red-400 mt-0.5" />
                  <p className="leading-snug">{errorMessage}</p>
                </div>
              )}

              <button
                onClick={handleUpload}
                disabled={!file || !apiUrl}
                className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 disabled:from-blue-950 disabled:to-slate-900 disabled:text-neutral-500 disabled:border-blue-900/30 border border-blue-400/40 text-white text-xs font-bold uppercase tracking-wider rounded-xl transition-all shadow-[0_0_20px_rgba(37,99,235,0.3)] flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
              >
                <Sparkles className="w-4 h-4" /> Generate 3D Reconstruction
              </button>
            </motion.div>
          ) : uploadState === "uploading" || uploadState === "processing" ? (
            <motion.div
              key="loading-view"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              className="flex flex-col items-center justify-center py-6 text-center"
            >
              <div className="relative w-14 h-14 mb-4 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border-3 border-blue-500/20 border-t-cyan-400 animate-spin" />
                <Server className="w-6 h-6 text-cyan-400" />
              </div>
              <h3 className="text-white font-bold text-sm">
                {uploadState === "uploading" ? "Streaming To Colab GPU" : "Running Metric Reconstruction"}
              </h3>
              <p className="text-blue-200/80 text-xs mt-1.5 px-4 font-mono">
                {pipelineStep}
              </p>
              <div className="mt-4 px-3 py-1 rounded-full bg-blue-950/80 border border-blue-500/30 text-[10px] text-cyan-300 font-mono flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />
                GPU Active • Do not close tab
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="success-view"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center justify-center py-6 text-center"
            >
              <div className="w-12 h-12 bg-emerald-500/20 rounded-full flex items-center justify-center mb-3 border border-emerald-500/40 text-emerald-400">
                <CheckCircle className="w-6 h-6" />
              </div>
              <h3 className="text-white font-bold text-sm">Model Built Successfully</h3>
              <p className="text-blue-300/70 text-xs mt-1">
                The 3D point cloud & mesh have been synthesized.
              </p>
              <button
                onClick={reset}
                className="mt-4 py-2 px-4 bg-blue-900/50 hover:bg-blue-800/60 border border-blue-500/30 text-blue-200 text-xs font-semibold rounded-lg transition-colors"
              >
                Upload Another Video
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      ) : (
        /* Direct / Demo Tab */
        <motion.div
          key="direct-panel"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col gap-4"
        >
          {/* Quick Demo Button */}
          <div className="p-4 bg-gradient-to-br from-blue-900/40 to-[#071329] border border-cyan-500/30 rounded-xl relative overflow-hidden">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/20 border border-cyan-400/40 text-cyan-300 shrink-0">
                <Layers className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <h4 className="text-xs font-bold text-white">Precomputed Digital Twin</h4>
                <p className="text-[11px] text-blue-200/70 mt-0.5 leading-relaxed">
                  Instantly preview the high-resolution 3D reconstruction with real geospatial telemetry without starting Colab.
                </p>
                <button
                  onClick={handleLoadSample}
                  className="mt-3 w-full py-2 px-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-bold rounded-lg transition-all shadow-md shadow-blue-500/20 flex items-center justify-center gap-1.5"
                >
                  Load Sample Model <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>

          {/* Direct URL Input */}
          <div className="flex flex-col gap-2 pt-2 border-t border-blue-500/20">
            <label className="text-xs font-semibold text-blue-200">
              Or Paste Existing Colab Model URL
            </label>
            <p className="text-[10px] text-blue-300/60 leading-tight">
              If your Colab has already generated a dataset, paste the endpoint directly to avoid waiting:
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="https://xxxx.loca.lt/model/dataset_..."
                value={directUrl}
                onChange={(e) => setDirectUrl(e.target.value)}
                className="flex-1 bg-[#060d1f]/90 border border-blue-500/30 rounded-xl px-3 py-2 text-xs text-white placeholder-blue-400/40 focus:outline-none focus:border-cyan-400 font-mono"
              />
              <button
                onClick={handleDirectSubmit}
                disabled={!directUrl.trim()}
                className="py-2 px-3.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white text-xs font-bold rounded-xl transition-all shrink-0"
              >
                Load
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
