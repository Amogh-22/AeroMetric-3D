"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
const ModelViewer = dynamic(() => import("@/components/ModelViewer"), { ssr: false });
import VideoUpload from "@/components/VideoUpload";
import { 
  Activity, 
  Map, 
  Compass, 
  Maximize2, 
  Navigation, 
  RotateCcw, 
  Layers, 
  Grid, 
  UploadCloud, 
  CheckCircle2, 
  Sparkles,
  Info,
  Radio,
  FileCode
} from "lucide-react";

export default function Home() {
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [geoMetadata, setGeoMetadata] = useState<any>(null);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // Check if there was a saved direct model URL in localStorage on mount
  useEffect(() => {
    try {
      const savedModel = localStorage.getItem("aerometric_active_model");
      if (savedModel && !savedModel.includes("loca.lt")) {
        // Only restore if not an ephemeral dead tunnel to prevent initial crash
        handleLoadModel(savedModel);
      }
    } catch (e) {}
  }, []);

  const handleLoadModel = async (url: string, customMetadataUrl?: string) => {
    setModelUrl(url);
    try {
      localStorage.setItem("aerometric_active_model", url);
    } catch (e) {}

    // Fetch metadata
    let metadataUrl = customMetadataUrl;
    if (!metadataUrl) {
      if (url.startsWith("/")) {
        metadataUrl = "/sample_metadata.json";
      } else {
        metadataUrl = url.replace("/model/", "/metadata/") + "?t=" + Date.now();
      }
    }

    try {
      const isLocal = metadataUrl.startsWith("/");
      const headers: Record<string, string> = isLocal ? {} : {
        "ngrok-skip-browser-warning": "true",
        "Bypass-Tunnel-Reminder": "true"
      };

      const res = await fetch(metadataUrl, {
        headers,
        cache: "no-store"
      });

      if (res.ok) {
        const data = await res.json();
        setGeoMetadata(data);
      } else {
        // Default telemetry fallback if metadata not yet created
        setGeoMetadata({
          epsg: 32643,
          anchor_lat: 12.9716,
          anchor_lon: 77.5946,
          anchor_alt: 50.2,
          num_points: 1312478
        });
      }
    } catch (e) {
      console.warn("Could not fetch metadata, using defaults:", e);
      setGeoMetadata({
        epsg: 32643,
        anchor_lat: 12.9716,
        anchor_lon: 77.5946,
        anchor_alt: 50.2,
        num_points: 1312478
      });
    }
  };

  const handleResetModel = () => {
    setModelUrl(null);
    setGeoMetadata(null);
    try {
      localStorage.removeItem("aerometric_active_model");
    } catch (e) {}
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  return (
    <main className="min-h-screen w-full bg-[#040816] bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(30,58,138,0.35),rgba(4,8,22,1))] text-blue-50 font-sans flex flex-col p-4 md:p-6 cyber-grid selection:bg-cyan-500/30 selection:text-white">
      
      {/* Top Header Bar */}
      <header className="w-full max-w-7xl mx-auto flex items-center justify-between pb-5 border-b border-blue-500/20 mb-6">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-[0_0_25px_rgba(56,189,248,0.4)] border border-cyan-400/40">
            <Radio className="w-6 h-6 text-white animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-xl md:text-2xl font-black tracking-tight text-white">
                AeroMetric<span className="text-cyan-400">-3D</span>
              </h1>
            </div>
            <p className="text-xs text-blue-300/70 font-mono flex items-center gap-1.5 mt-0.5">
              <span>DRONE PHOTOGRAMMETRY & DIGITAL TWIN WORKSPACE</span>
            </p>
          </div>
        </div>

        {/* Header Actions */}
        <div className="flex items-center gap-3">
          {modelUrl ? (
            <button
              onClick={handleResetModel}
              className="px-3.5 py-1.5 rounded-xl bg-blue-950/80 hover:bg-blue-900/80 border border-blue-500/30 text-xs font-semibold text-blue-200 transition-colors flex items-center gap-2"
            >
              <UploadCloud className="w-3.5 h-3.5 text-cyan-400" />
              <span>Change Video / Model</span>
            </button>
          ) : (
            <button
              onClick={() => handleLoadModel("/sample_model.glb", "/sample_metadata.json")}
              className="px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-xs font-bold text-white transition-all shadow-[0_0_20px_rgba(37,99,235,0.3)] flex items-center gap-2"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Quick Demo</span>
            </button>
          )}

          <button
            onClick={toggleFullscreen}
            title="Toggle Fullscreen"
            className="p-2 rounded-xl bg-[#0a152e]/80 hover:bg-blue-600/20 border border-blue-500/30 text-blue-300 hover:text-white transition-colors"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="w-full max-w-7xl mx-auto flex-1 flex flex-col lg:flex-row gap-6 items-start justify-center">
        
        {/* 3D Model Bounded Viewport (Strictly inside a boundary box) */}
        <div className="w-full lg:flex-1 h-[65vh] lg:h-[75vh] relative rounded-3xl border border-blue-500/30 bg-[#060e22]/90 shadow-[0_0_50px_rgba(30,58,138,0.25)] backdrop-blur-2xl flex flex-col overflow-hidden">
          
          {/* Viewport Top Bar */}
          <div className="h-12 px-5 bg-[#091530]/80 border-b border-blue-500/20 flex items-center justify-between z-10 select-none">
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
              <span className="text-xs font-bold text-white tracking-wide uppercase">
                {modelUrl ? "3D Mesh Viewport" : "Viewport Standby"}
              </span>
              <span className="text-[10px] font-mono text-blue-400/80 px-2 py-0.5 rounded bg-blue-950/60 border border-blue-500/20 hidden sm:inline-block">
                EPSG:32643 • WGS84 UTM
              </span>
            </div>

            {/* Quick Viewport Controls */}
            {modelUrl && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowGrid(!showGrid)}
                  title="Toggle Reference Grid"
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                    showGrid 
                      ? "bg-blue-600 text-white shadow-sm shadow-blue-500/30" 
                      : "bg-blue-950/60 text-blue-300/70 hover:text-white border border-blue-500/20"
                  }`}
                >
                  <Grid className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Grid</span>
                </button>

                <button
                  onClick={() => setWireframe(!wireframe)}
                  title="Toggle Wireframe Mode"
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                    wireframe 
                      ? "bg-cyan-500 text-slate-950 font-bold shadow-sm shadow-cyan-500/30" 
                      : "bg-blue-950/60 text-blue-300/70 hover:text-white border border-blue-500/20"
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Wireframe</span>
                </button>
              </div>
            )}
          </div>

          {/* Interactive 3D Canvas Area */}
          <div className="relative flex-1 w-full h-full overflow-hidden">
            {modelUrl ? (
              <ModelViewer 
                modelUrl={modelUrl}
                wireframe={wireframe}
                showGrid={showGrid}
                onReset={handleResetModel}
                onLoadFallback={() => handleLoadModel("/sample_model.glb", "/sample_metadata.json")}
              />
            ) : (
              /* Awaiting Model Standby State */
              <div className="absolute inset-0 w-full h-full flex flex-col items-center justify-center p-6 text-center select-none">
                {/* Tech Radar Circle Graphic */}
                <div className="relative w-36 h-36 mb-6 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border border-blue-500/20 animate-[spin_12s_linear_infinite]" />
                  <div className="absolute inset-3 rounded-full border border-dashed border-cyan-400/30 animate-[spin_20s_linear_infinite_reverse]" />
                  <div className="absolute inset-8 rounded-full border border-blue-500/40" />
                  <div className="w-12 h-12 rounded-2xl bg-blue-600/20 border border-blue-400/40 flex items-center justify-center text-cyan-300 shadow-[0_0_30px_rgba(56,189,248,0.3)]">
                    <Compass className="w-6 h-6 animate-pulse" />
                  </div>
                </div>

                <h3 className="text-lg md:text-xl font-bold text-white tracking-wide">
                  AeroMetric Viewport Ready
                </h3>
                <p className="text-xs md:text-sm text-blue-200/70 max-w-md mt-2 leading-relaxed">
                  Upload drone footage on the control panel to synthesize a geo-referenced 3D digital twin, or launch the demo reconstruction immediately.
                </p>

                <div className="mt-6 flex flex-wrap gap-3 items-center justify-center">
                  <button
                    onClick={() => handleLoadModel("/sample_model.glb", "/sample_metadata.json")}
                    className="px-5 py-2.5 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white text-xs font-bold rounded-xl transition-all shadow-[0_0_25px_rgba(37,99,235,0.4)] flex items-center gap-2 cursor-pointer"
                  >
                    <Sparkles className="w-4 h-4" /> Load Sample Digital Twin
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Corner Aerospace Bracket Accents */}
          <div className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-cyan-400/60 pointer-events-none" />
          <div className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-cyan-400/60 pointer-events-none" />
          <div className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-cyan-400/60 pointer-events-none" />
          <div className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-cyan-400/60 pointer-events-none" />
        </div>

        {/* Right Side Panel: Controls & Telemetry */}
        <aside className="w-full lg:w-[380px] flex flex-col gap-4 shrink-0">
          
          {/* If No Model Loaded: Show Upload / Direct URL panel */}
          {!modelUrl ? (
            <>
              <VideoUpload 
                onUploadComplete={(url) => handleLoadModel(url)}
                onLoadDirectModel={(url, meta) => handleLoadModel(url, meta)}
              />

              {/* Architecture Info Card */}
              <div className="p-5 bg-[#0a152e]/85 backdrop-blur-xl border border-blue-500/20 rounded-2xl shadow-xl flex flex-col gap-3">
                <div className="flex items-center gap-2 text-cyan-400 text-xs font-bold uppercase tracking-wider">
                  <Info className="w-4 h-4" /> Pipeline Capabilities
                </div>
                <ul className="text-xs text-blue-200/70 space-y-2 font-mono">
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    Deep Metric Depth & TSDF Fusion
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    Screened Poisson Surface Meshing
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    Automatic UTM Coordinate Normalization
                  </li>
                </ul>
              </div>
            </>
          ) : (
            /* If Model Is Loaded: Show Telemetry & Controls */
            <>
              {/* Telemetry Card */}
              <div className="p-5 bg-[#0a152e]/90 backdrop-blur-xl border border-blue-500/25 rounded-2xl shadow-xl flex flex-col gap-5">
                
                {/* Header */}
                <div className="flex items-center justify-between pb-3 border-b border-blue-500/20">
                  <div className="flex items-center gap-2 text-white font-bold text-sm">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    <span>Reconstruction Telemetry</span>
                  </div>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px] font-bold">
                    ONLINE
                  </span>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-[#060e22] border border-blue-500/20 flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-blue-300/70">Estimated Points</span>
                    <span className="text-lg font-bold font-mono text-white">
                      {geoMetadata?.num_points ? (geoMetadata.num_points / 1e6).toFixed(2) + 'M' : '1.31M'}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#060e22] border border-blue-500/20 flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-blue-300/70">Triangles / Faces</span>
                    <span className="text-lg font-bold font-mono text-cyan-400">
                      {geoMetadata?.num_points ? ((geoMetadata.num_points * 2) / 1e6).toFixed(2) + 'M' : '2.62M'}
                    </span>
                  </div>
                </div>

                {/* Geospatial Anchor Info */}
                <div className="flex flex-col gap-2.5 pt-2 border-t border-blue-500/20">
                  <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider mb-1">
                    <Map className="w-3.5 h-3.5 text-blue-400" />
                    <span>Geospatial Anchor</span>
                  </div>

                  <div className="flex justify-between items-center text-xs py-1 px-2.5 rounded-lg bg-[#060e22]/60">
                    <span className="text-blue-300/70">CRS Projection</span>
                    <span className="font-mono text-white font-semibold">EPSG:{geoMetadata?.epsg ?? 32643}</span>
                  </div>

                  <div className="flex justify-between items-center text-xs py-1 px-2.5 rounded-lg bg-[#060e22]/60">
                    <span className="text-blue-300/70">Latitude</span>
                    <span className="font-mono text-cyan-300">
                      {geoMetadata?.anchor_lat !== undefined ? `${geoMetadata.anchor_lat.toFixed(6)}° N` : '12.971600° N'}
                    </span>
                  </div>

                  <div className="flex justify-between items-center text-xs py-1 px-2.5 rounded-lg bg-[#060e22]/60">
                    <span className="text-blue-300/70">Longitude</span>
                    <span className="font-mono text-cyan-300">
                      {geoMetadata?.anchor_lon !== undefined ? `${geoMetadata.anchor_lon.toFixed(6)}° E` : '77.594600° E'}
                    </span>
                  </div>

                  <div className="flex justify-between items-center text-xs py-1 px-2.5 rounded-lg bg-[#060e22]/60">
                    <span className="text-blue-300/70">Base Altitude</span>
                    <span className="font-mono text-emerald-300">
                      {geoMetadata?.anchor_alt !== undefined ? `${geoMetadata.anchor_alt.toFixed(1)} m` : '50.2 m'}
                    </span>
                  </div>

                  {geoMetadata?.utm_x !== undefined && (
                    <div className="flex justify-between items-center text-xs py-1 px-2.5 rounded-lg bg-[#060e22]/60">
                      <span className="text-blue-300/70">UTM Coordinates</span>
                      <span className="font-mono text-blue-200 text-[11px]">
                        X: {Math.round(geoMetadata.utm_x)} | Y: {Math.round(geoMetadata.utm_y)}
                      </span>
                    </div>
                  )}
                </div>

                {/* Reset / New Video Button */}
                <button
                  onClick={handleResetModel}
                  className="w-full py-2.5 px-4 bg-blue-900/40 hover:bg-blue-800/50 border border-blue-500/30 text-blue-200 text-xs font-semibold rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  <UploadCloud className="w-4 h-4 text-cyan-400" />
                  Process Another Video
                </button>
              </div>

              {/* Navigation Hints Card */}
              <div className="p-4 bg-[#0a152e]/80 backdrop-blur-xl border border-blue-500/20 rounded-2xl flex items-center gap-3.5 shadow-lg">
                <div className="p-2.5 rounded-xl bg-blue-600/20 border border-blue-500/30 text-cyan-400 shrink-0">
                  <Navigation className="w-5 h-5" />
                </div>
                <div className="text-xs text-blue-200/80 leading-relaxed">
                  <p><span className="text-white font-bold">Left Click + Drag</span> to orbit 3D view.</p>
                  <p><span className="text-white font-bold">Scroll Wheel</span> to zoom in/out.</p>
                  <p><span className="text-white font-bold">Right Click + Drag</span> to pan camera.</p>
                </div>
              </div>
            </>
          )}

        </aside>

      </div>

    </main>
  );
}
