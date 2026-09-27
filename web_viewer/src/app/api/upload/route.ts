import { NextRequest, NextResponse } from "next/server";
const archiver = require('archiver');
import fs from 'fs';
import path from 'path';

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const formData = await req.formData();
    const file = formData.get("video") as File;
    const ngrokUrl = formData.get("ngrokUrl") as string;
    
    if (!file) return NextResponse.json({ error: "No video file provided" }, { status: 400 });
    if (!ngrokUrl) return NextResponse.json({ error: "No Ngrok URL provided" }, { status: 400 });

    const buffer = Buffer.from(await file.arrayBuffer());
    
    console.log("Preparing zip file...");
    // Zip python files in the parent directory
    const scriptsDir = path.join(process.cwd(), '..');
    
    return new Promise<NextResponse>((resolve) => {
      const archive = new archiver.ZipArchive({ zlib: { level: 9 } });
      const bufs: any[] = [];
      
      archive.on('data', function(data: Buffer) {
        bufs.push(data);
      });
      
      archive.on('end', async function() {
        const zipBuffer = Buffer.concat(bufs);
        
        console.log("Sending to GPU Server:", ngrokUrl);
        const datasetName = `dataset_${Date.now()}`;
        const gpuFormData = new FormData();
        
        const videoBlob = new Blob([buffer], { type: 'video/mp4' });
        const zipBlob = new Blob([zipBuffer], { type: 'application/zip' });
        
        // Sanitize video filename to eliminate spaces and special shell characters
        const safeFileName = (file.name || "video.mp4").replace(/[^a-zA-Z0-9._-]/g, "_");
        
        gpuFormData.append("video", videoBlob, safeFileName);
        gpuFormData.append("scripts_zip", zipBlob, "scripts.zip");
        gpuFormData.append("dataset_name", datasetName);
        
        try {
          const response = await fetch(`${ngrokUrl}/upload`, {
            method: 'POST',
            body: gpuFormData,
            headers: {
              "ngrok-skip-browser-warning": "true",
              "Bypass-Tunnel-Reminder": "true"
            }
          });
          
          if (!response.ok) {
            resolve(NextResponse.json({ error: await response.text() }, { status: response.status }));
            return;
          }
          
          const data = await response.json();
          resolve(NextResponse.json({ success: true, datasetName: datasetName }));
        } catch (error: any) {
          resolve(NextResponse.json({ error: error.message }, { status: 500 }));
        }
      });
      
      archive.on('error', function(err: any) {
        resolve(NextResponse.json({ error: err.message }, { status: 500 }));
      });
      
      // Append python files
      ['generate_mesh.py', 'metric_depth_fusion.py', 'preprocess_drone_video.py', 'run_pipeline.py', 'run_sfm.py', 'parse_telemetry.py', 'geospatial_export.py'].forEach(pyFile => {
        const p = path.join(/*turbopackIgnore: true*/ scriptsDir, pyFile);
        if (fs.existsSync(/*turbopackIgnore: true*/ p)) {
          archive.file(p, { name: pyFile });
        }
      });
      
      archive.finalize();
    });

  } catch (error: any) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
