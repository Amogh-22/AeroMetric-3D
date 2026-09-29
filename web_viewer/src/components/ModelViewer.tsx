"use client";

import React, { useEffect, useState, useRef, useMemo } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Html, useProgress } from "@react-three/drei";
import { AlertCircle, RotateCcw, Box, RefreshCw, Eye, ArrowUpDown, Compass } from "lucide-react";

function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="flex flex-col items-center gap-3 bg-[#0a1428]/90 border border-blue-500/30 px-6 py-4 rounded-2xl shadow-[0_0_30px_rgba(59,130,246,0.3)] backdrop-blur-xl text-center">
        <div className="relative w-10 h-10 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border-2 border-blue-500/20 border-t-cyan-400 animate-spin" />
          <Box className="w-5 h-5 text-cyan-400 animate-pulse" />
        </div>
        <div className="text-white text-sm font-semibold tracking-wide">
          Streaming Mesh Geometry
        </div>
        <div className="w-40 bg-blue-950/80 rounded-full h-1.5 overflow-hidden border border-blue-500/30">
          <div 
            className="bg-gradient-to-r from-blue-500 to-cyan-400 h-full transition-all duration-300"
            style={{ width: `${Math.max(progress, 5)}%` }}
          />
        </div>
        <span className="text-[11px] font-mono text-cyan-300">
          {progress > 0 ? `${progress.toFixed(0)}%` : "Initializing..."}
        </span>
      </div>
    </Html>
  );
}

function MeshModel({ 
  url, 
  wireframe = false,
  invertY = false,
  showBoundingBox = true
}: { 
  url: string; 
  wireframe?: boolean;
  invertY?: boolean;
  showBoundingBox?: boolean;
}) {
  const { scene } = useGLTF(url);
  
  // Clone scene so we never mutate the useGLTF cache, ensuring stable scale & position across reloads
  const { modelGroup, center, scale, size } = useMemo(() => {
    const clone = scene.clone(true);
    
    // Compute exact bounding box on pristine unscaled geometry
    const box = new THREE.Box3().setFromObject(clone);
    const c = box.getCenter(new THREE.Vector3());
    const sz = box.getSize(new THREE.Vector3());
    
    const maxDim = Math.max(sz.x, sz.y, sz.z);
    const sc = maxDim > 0 ? 10.5 / maxDim : 1;

    // Initialize materials once with pure white color (0xffffff)
    // so original vertex colors and textures render with 100% natural photographic fidelity
    clone.traverse((child: any) => {
      if (child.isMesh && child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        
        const newMaterials = materials.map((oldMat: any) => {
          const hasVertexColors = Boolean(
            child.geometry && 
            child.geometry.attributes && 
            child.geometry.attributes.color !== undefined
          );

          return new THREE.MeshStandardMaterial({
            color: new THREE.Color(0xffffff), // PURE WHITE: Never tint vertex colors!
            map: oldMat.map || null,
            vertexColors: hasVertexColors,
            roughness: 0.8,
            metalness: 0.1,
            wireframe: false,
            side: THREE.DoubleSide
          });
        });

        child.material = Array.isArray(child.material) ? newMaterials : newMaterials[0];
      }
    });

    return {
      modelGroup: clone,
      center: c,
      scale: sc,
      size: sz
    };
  }, [scene]);

  // Update wireframe property in-place on child materials without replacing them or mutating colors
  useEffect(() => {
    if (!modelGroup) return;
    modelGroup.traverse((child: any) => {
      if (child.isMesh && child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach((m: any) => {
          m.wireframe = Boolean(wireframe);
          m.needsUpdate = true;
        });
      }
    });
  }, [modelGroup, wireframe]);

  return (
    <group 
      scale={[scale, invertY ? -scale : scale, scale]} 
      position={[-center.x * scale, -center.y * (invertY ? -scale : scale), -center.z * scale]}
      rotation={[invertY ? Math.PI : 0, 0, 0]}
    >
      <primitive object={modelGroup} />

      {/* Holographic Bounding Box Cage enclosing the 3D model */}
      {showBoundingBox && size && (
        <mesh position={[center.x, center.y, center.z]}>
          <boxGeometry args={[size.x * 1.02, size.y * 1.02, size.z * 1.02]} />
          <meshBasicMaterial 
            color="#38bdf8" 
            wireframe 
            transparent 
            opacity={0.35} 
          />
        </mesh>
      )}
    </group>
  );
}

interface ModelViewerProps {
  modelUrl: string | null;
  onReset?: () => void;
  onLoadFallback?: () => void;
  wireframe?: boolean;
  showGrid?: boolean;
}

export default function ModelViewer({ 
  modelUrl, 
  onReset, 
  onLoadFallback,
  wireframe = false,
  showGrid = true
}: ModelViewerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<string>("0 MB");
  const [invertY, setInvertY] = useState(false);
  const [showBoundingBox, setShowBoundingBox] = useState(true);
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    if (!modelUrl) {
      setBlobUrl(null);
      setErrorMsg(null);
      return;
    }
    
    let isMounted = true;
    setErrorMsg(null);
    setIsDownloading(true);
    
    // If it's already a relative/local URL, use it directly
    if (modelUrl.startsWith("/")) {
      setBlobUrl(modelUrl);
      setIsDownloading(false);
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout

    fetch(modelUrl, {
      headers: { 
        "ngrok-skip-browser-warning": "true",
        "Bypass-Tunnel-Reminder": "true" 
      },
      signal: controller.signal
    })
      .then(async (res) => {
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to reach GPU backend (${res.statusText})`);
        
        const contentLength = res.headers.get("content-length");
        if (contentLength && isMounted) {
          const mb = (parseInt(contentLength, 10) / (1024 * 1024)).toFixed(1);
          setDownloadProgress(`${mb} MB`);
        }
        return res.blob();
      })
      .then(blob => {
        if (isMounted) {
          setBlobUrl(URL.createObjectURL(blob));
          setIsDownloading(false);
        }
      })
      .catch(err => {
        clearTimeout(timeoutId);
        console.error("Failed to fetch model blob:", err);
        if (isMounted) {
          setIsDownloading(false);
          if (err.name === 'AbortError') {
            setErrorMsg("Connection timed out. The Colab/Ngrok tunnel may be inactive or expired.");
          } else {
            setErrorMsg(err.message || "Failed to connect to backend endpoint");
          }
        }
      });

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [modelUrl]);

  const handleResetCamera = () => {
    if (controlsRef.current) {
      controlsRef.current.reset();
      controlsRef.current.target.set(0, 0, 0);
    }
  };

  const setCameraView = (view: 'iso' | 'front' | 'top' | 'bottom') => {
    if (!controlsRef.current) return;
    const ctrl = controlsRef.current;
    if (view === 'iso') {
      ctrl.object.position.set(6.5, 4.5, 7.5);
    } else if (view === 'front') {
      ctrl.object.position.set(0, 1.2, 9.5);
    } else if (view === 'top') {
      ctrl.object.position.set(0, 11.0, 0.01);
    } else if (view === 'bottom') {
      ctrl.object.position.set(0, -11.0, 0.01);
    }
    ctrl.target.set(0, 0, 0);
    ctrl.update();
  };

  if (!modelUrl) return null;

  if (errorMsg) {
    return (
      <div className="absolute inset-0 w-full h-full bg-[#070e1e] flex items-center justify-center p-6">
        <div className="max-w-md w-full p-6 bg-[#0c1935]/95 border border-red-500/40 rounded-2xl shadow-[0_0_50px_rgba(239,68,68,0.2)] backdrop-blur-xl flex flex-col items-center text-center gap-4">
          <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400">
            <AlertCircle className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-white font-bold text-lg mb-1">Backend Connection Error</h3>
            <p className="text-red-300/80 text-xs leading-relaxed font-mono px-2 py-1 bg-red-950/40 rounded-lg border border-red-500/20">
              {errorMsg}
            </p>
          </div>
          <p className="text-neutral-400 text-xs">
            The previous Colab session or tunnel URL has expired. You can upload a new video with a fresh tunnel or preview the local sample model.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 w-full mt-2">
            {onLoadFallback && (
              <button
                onClick={onLoadFallback}
                className="flex-1 py-2.5 px-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-bold rounded-xl transition-all shadow-[0_0_20px_rgba(37,99,235,0.3)] flex items-center justify-center gap-2"
              >
                <Eye className="w-4 h-4" /> Load Sample Model
              </button>
            )}
            {onReset && (
              <button
                onClick={onReset}
                className="flex-1 py-2.5 px-4 bg-blue-950/60 hover:bg-blue-900/60 border border-blue-500/30 text-blue-200 text-xs font-semibold rounded-xl transition-colors flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-4 h-4" /> Upload Video
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (isDownloading || !blobUrl) {
    return (
      <div className="absolute inset-0 w-full h-full bg-[#070e1e] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 bg-[#0a1428]/90 border border-blue-500/30 px-8 py-6 rounded-2xl shadow-[0_0_40px_rgba(59,130,246,0.25)] backdrop-blur-xl text-center">
          <div className="relative w-12 h-12 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border-3 border-blue-500/20 border-t-cyan-400 animate-spin" />
            <Box className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h4 className="text-white font-semibold text-base">Streaming 3D Digital Twin</h4>
            <p className="text-blue-300/70 text-xs mt-1">
              Fetching metric reconstruction from Colab GPU ({downloadProgress})
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full bg-gradient-to-b from-[#081226] via-[#060e1d] to-[#040914] overflow-hidden select-none">
      {/* Floating Viewport Quick Controls Toolbar */}
      <div className="absolute top-4 right-4 z-10 flex flex-wrap items-center gap-2 bg-[#09152e]/85 backdrop-blur-md border border-blue-500/25 rounded-xl p-1.5 shadow-[0_4px_25px_rgba(0,0,0,0.4)]">
        {/* Flip 180° / Invert Y Axis button */}
        <button
          onClick={() => setInvertY(prev => !prev)}
          title="Flip Model 180° Upright"
          className={`p-1.5 rounded-lg transition-all flex items-center gap-1.5 text-xs font-medium ${
            invertY 
              ? "bg-cyan-500/30 text-cyan-200 border border-cyan-400/40 shadow-[0_0_10px_rgba(6,182,212,0.3)]" 
              : "text-blue-300 hover:text-white hover:bg-blue-600/30"
          }`}
        >
          <ArrowUpDown className="w-3.5 h-3.5" />
          <span className="text-[11px] hidden sm:inline">Flip 180°</span>
        </button>

        {/* Boundary Box Cage Toggle */}
        <button
          onClick={() => setShowBoundingBox(prev => !prev)}
          title="Toggle 3D Boundary Box Cage"
          className={`p-1.5 rounded-lg transition-all flex items-center gap-1.5 text-xs font-medium ${
            showBoundingBox 
              ? "bg-blue-600/30 text-blue-200 border border-blue-400/30 shadow-[0_0_10px_rgba(59,130,246,0.2)]" 
              : "text-blue-400/60 hover:text-white hover:bg-blue-600/20"
          }`}
        >
          <Box className="w-3.5 h-3.5" />
          <span className="text-[11px] hidden sm:inline">Box Cage</span>
        </button>

        <div className="h-4 w-[1px] bg-blue-500/30 mx-0.5" />

        {/* Camera Preset Angles */}
        <button
          onClick={() => setCameraView('front')}
          title="Front View"
          className="px-2 py-1 text-[11px] font-medium text-blue-300 hover:text-white hover:bg-blue-600/30 rounded-lg transition-colors"
        >
          Front
        </button>
        <button
          onClick={() => setCameraView('top')}
          title="Top View"
          className="px-2 py-1 text-[11px] font-medium text-blue-300 hover:text-white hover:bg-blue-600/30 rounded-lg transition-colors"
        >
          Top
        </button>
        <button
          onClick={() => setCameraView('bottom')}
          title="Bottom View (Inspect Foundation)"
          className="px-2 py-1 text-[11px] font-medium text-cyan-300 hover:text-white hover:bg-cyan-600/30 rounded-lg transition-colors"
        >
          Bottom
        </button>

        <div className="h-4 w-[1px] bg-blue-500/30 mx-0.5" />

        {/* Reset Camera */}
        <button
          onClick={handleResetCamera}
          title="Reset Camera View"
          className="p-1.5 text-blue-300 hover:text-white hover:bg-blue-600/30 rounded-lg transition-colors flex items-center gap-1 text-xs"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="text-[11px] font-medium hidden sm:inline">Reset</span>
        </button>
      </div>

      {/* 3D Canvas */}
      <Canvas
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#060e1d', width: '100%', height: '100%' }}
        camera={{ position: [0, 4.0, 8.8], fov: 45 }}
      >
        <color attach="background" args={['#060e1d']} />
        <fog attach="fog" args={['#060e1d', 30, 100]} />

        <ambientLight intensity={1.5} />
        <directionalLight position={[15, 25, 20]} intensity={2.2} />
        <directionalLight position={[-15, -15, -15]} intensity={1.1} color="#60a5fa" />
        
        {/* Aerospace Coordinate Reference Grid right beneath building base */}
        {showGrid && (
          <gridHelper 
            args={[24, 24, 0x38bdf8, 0x1e3a8a]} 
            position={[0, -0.9, 0]} 
          />
        )}
        
        <React.Suspense fallback={<Loader />}>
          <MeshModel 
            key={blobUrl} 
            url={blobUrl} 
            wireframe={wireframe} 
            invertY={invertY}
            showBoundingBox={showBoundingBox}
          />
        </React.Suspense>

        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.06}
          minDistance={0.5}
          maxDistance={150}
          minPolarAngle={0}
          maxPolarAngle={Math.PI}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  );
}
