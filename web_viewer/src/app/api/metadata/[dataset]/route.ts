import { NextRequest, NextResponse } from "next/server";
import { join } from "path";
import fs from "fs";

export async function GET(req: NextRequest, { params }: { params: Promise<{ dataset: string }> }) {
  const { dataset: datasetName } = await params;
  
  // The dataset folder is in TEST1
  const projectRoot = join(process.cwd(), "..");
  const geoPath = join(projectRoot, datasetName, "geospatial_metadata.json");

  if (!fs.existsSync(geoPath)) {
    return NextResponse.json({ error: "Metadata not ready" }, { status: 404 });
  }

  const fileData = fs.readFileSync(geoPath, 'utf8');
  return NextResponse.json(JSON.parse(fileData));
}
