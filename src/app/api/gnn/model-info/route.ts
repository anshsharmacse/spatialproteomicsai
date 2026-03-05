import { NextResponse } from 'next/server';

export async function GET() {
  // Return model info directly (works on Vercel)
  return NextResponse.json({
    architecture: {
      type: 'Message-Passing Neural Network',
      hiddenDim: 64,
      numLayers: 3,
      dropout: 0.2,
      activation: 'ReLU'
    },
    features: [
      'Spatial coordinates (normalized)',
      'Protein expression levels',
      'Distance-based features',
      'Angular features',
      'Log-transformed expression'
    ],
    performance: {
      accuracy: 0.82,
      task: 'Protein Subcellular Localization Prediction'
    },
    capabilities: [
      'Spatial adjacency graph construction',
      'Message passing for feature aggregation',
      'Protein localization prediction',
      'RNA-seq integration',
      'Cell type deconvolution'
    ]
  });
}
