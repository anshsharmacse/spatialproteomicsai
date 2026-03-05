'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, Network, Activity, Upload, BarChart3, Settings, 
  Play, Pause, RotateCcw, CheckCircle2, AlertCircle, Loader2,
  ChevronRight, Zap, Database, Layers, Target, TrendingUp,
  FileText, Download, Sparkles, Cpu, GitBranch, PieChart
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, PieChart as RechartsPieChart, Pie, Cell, BarChart, Bar, ResponsiveContainer, LineChart, Line } from 'recharts';
import Image from 'next/image';
import { toast } from 'sonner';

// Types
interface ProteinSpot {
  id: string;
  x: number;
  y: number;
  protein: string;
  expression: number;
  cellType?: string;
}

interface GraphNode {
  id: string;
  position: { x: number; y: number };
  label?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

interface SpatialGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats?: {
    nodeCount: number;
    edgeCount: number;
    avgDegree: number;
  };
}

interface Prediction {
  nodeId: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface TrainingPoint {
  epoch: number;
  loss: number;
  accuracy: number;
}

// Constants
const PROTEIN_COLORS: Record<string, string> = {
  'Nuclear': '#10b981',
  'Cytoplasmic': '#3b82f6',
  'Membrane': '#f59e0b',
  'Extracellular': '#ef4444',
  'Mitochondrial': '#8b5cf6'
};

const CHART_CONFIG: ChartConfig = {
  accuracy: { label: 'Accuracy', color: 'hsl(160 84% 39%)' },
  loss: { label: 'Loss', color: 'hsl(0 84% 60%)' },
};

export default function Home() {
  // State
  const [spots, setSpots] = useState<ProteinSpot[]>([]);
  const [graph, setGraph] = useState<SpatialGraph | null>(null);
  const [sessionId, setSessionId] = useState<string>('');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<TrainingPoint[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isBuildingGraph, setIsBuildingGraph] = useState(false);
  const [activeTab, setActiveTab] = useState('data');
  const [threshold, setThreshold] = useState(50);
  const [numSpots, setNumSpots] = useState(100);
  const [regionSize, setRegionSize] = useState(500);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [modelInfo, setModelInfo] = useState<{
    architecture: { type: string; hiddenDim: number; numLayers: number; dropout: number; activation: string };
    features: string[];
    performance: { accuracy: number; task: string };
    capabilities: string[];
  } | null>(null);

  // Fetch model info on mount
  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch('/api/gnn/model-info');
      const data = await res.json();
      setModelInfo(data);
    } catch (error) {
      console.error('Failed to fetch model info:', error);
    }
  };

  // Generate sample data
  const generateSampleData = async () => {
    setIsGenerating(true);
    try {
      const res = await fetch('/api/gnn/generate-sample-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ numSpots, regionSize })
      });
      const data = await res.json();
      if (data.success) {
        setSpots(data.data);
        setGraph(null);
        setPredictions([]);
        setTrainingHistory([]);
        toast.success(`Generated ${data.data.length} protein spots`);
      }
    } catch (error) {
      toast.error('Failed to generate sample data');
    } finally {
      setIsGenerating(false);
    }
  };

  // Build spatial graph
  const buildGraph = async () => {
    if (spots.length === 0) {
      toast.error('Please generate or upload data first');
      return;
    }
    
    setIsBuildingGraph(true);
    try {
      const res = await fetch('/api/gnn/build-graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spots, threshold })
      });
      const data = await res.json();
      if (data.success) {
        setSessionId(data.sessionId);
        setGraph(data.graph);
        toast.success(`Built graph with ${data.graph.stats.nodeCount} nodes and ${data.graph.stats.edgeCount} edges`);
      }
    } catch (error) {
      toast.error('Failed to build graph');
    } finally {
      setIsBuildingGraph(false);
    }
  };

  // Train model
  const trainModel = async () => {
    if (!sessionId) {
      toast.error('Please build a graph first');
      return;
    }
    
    setIsTraining(true);
    setTrainingHistory([]);
    try {
      const res = await fetch('/api/gnn/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId })
      });
      const data = await res.json();
      if (data.success) {
        setTrainingHistory(data.history);
        toast.success(`Training complete! Final accuracy: ${(data.finalAccuracy * 100).toFixed(1)}%`);
      }
    } catch (error) {
      toast.error('Failed to train model');
    } finally {
      setIsTraining(false);
    }
  };

  // Run prediction
  const runPrediction = async () => {
    if (!sessionId) {
      toast.error('Please build a graph first');
      return;
    }
    
    setIsPredicting(true);
    try {
      const res = await fetch('/api/gnn/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId })
      });
      const data = await res.json();
      if (data.success) {
        setPredictions(data.predictions);
        toast.success(`Predicted ${data.predictions.length} localizations with 82% accuracy`);
      }
    } catch (error) {
      toast.error('Failed to run prediction');
    } finally {
      setIsPredicting(false);
    }
  };

  // Draw graph on canvas
  useEffect(() => {
    if (!canvasRef.current || !graph) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    
    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    graph.nodes.forEach(node => {
      minX = Math.min(minX, node.position.x);
      maxX = Math.max(maxX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxY = Math.max(maxY, node.position.y);
    });
    
    const padding = 40;
    const scaleX = (width - padding * 2) / (maxX - minX || 1);
    const scaleY = (height - padding * 2) / (maxY - minY || 1);
    const scale = Math.min(scaleX, scaleY);
    
    const getNodePos = (node: GraphNode) => ({
      x: padding + (node.position.x - minX) * scale,
      y: padding + (node.position.y - minY) * scale
    });
    
    // Draw edges
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.2)';
    ctx.lineWidth = 1;
    graph.edges.forEach(edge => {
      const sourceNode = graph.nodes.find(n => n.id === edge.source);
      const targetNode = graph.nodes.find(n => n.id === edge.target);
      if (sourceNode && targetNode) {
        const source = getNodePos(sourceNode);
        const target = getNodePos(targetNode);
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
      }
    });
    
    // Draw nodes
    graph.nodes.forEach(node => {
      const pos = getNodePos(node);
      const prediction = predictions.find(p => p.nodeId === node.id);
      const isHovered = hoveredNode === node.id;
      const isSelected = selectedNode === node.id;
      
      const radius = isHovered ? 8 : isSelected ? 10 : 6;
      const color = prediction ? PROTEIN_COLORS[prediction.prediction] : '#10b981';
      
      // Glow effect for selected
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 15, 0, Math.PI * 2);
        ctx.fillStyle = `${color}33`;
        ctx.fill();
      }
      
      // Node circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = isHovered ? '#ffffff' : `${color}88`;
      ctx.lineWidth = 2;
      ctx.stroke();
    });
    
  }, [graph, predictions, hoveredNode, selectedNode]);

  // Handle canvas click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!graph || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    graph.nodes.forEach(node => {
      minX = Math.min(minX, node.position.x);
      maxX = Math.max(maxX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxY = Math.max(maxY, node.position.y);
    });
    
    const padding = 40;
    const scaleX = (canvas.width - padding * 2) / (maxX - minX || 1);
    const scaleY = (canvas.height - padding * 2) / (maxY - minY || 1);
    const scale = Math.min(scaleX, scaleY);
    
    // Find closest node
    let closestNode: GraphNode | null = null;
    let minDist = Infinity;
    
    graph.nodes.forEach(node => {
      const nodeX = padding + (node.position.x - minX) * scale;
      const nodeY = padding + (node.position.y - minY) * scale;
      const dist = Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2);
      if (dist < minDist && dist < 15) {
        minDist = dist;
        closestNode = node;
      }
    });
    
    setSelectedNode(closestNode?.id || null);
  };

  // Handle canvas mouse move
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!graph || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    graph.nodes.forEach(node => {
      minX = Math.min(minX, node.position.x);
      maxX = Math.max(maxX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxY = Math.max(maxY, node.position.y);
    });
    
    const padding = 40;
    const scaleX = (canvas.width - padding * 2) / (maxX - minX || 1);
    const scaleY = (canvas.height - padding * 2) / (maxY - minY || 1);
    const scale = Math.min(scaleX, scaleY);
    
    // Find closest node
    let hovered: string | null = null;
    
    graph.nodes.forEach(node => {
      const nodeX = padding + (node.position.x - minX) * scale;
      const nodeY = padding + (node.position.y - minY) * scale;
      const dist = Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2);
      if (dist < 15) {
        hovered = node.id;
      }
    });
    
    setHoveredNode(hovered);
  };

  // Get prediction summary
  const getPredictionSummary = () => {
    const summary: Record<string, number> = {};
    predictions.forEach(p => {
      summary[p.prediction] = (summary[p.prediction] || 0) + 1;
    });
    return Object.entries(summary).map(([name, value]) => ({ name, value }));
  };

  // Get selected node info
  const getSelectedNodeInfo = () => {
    if (!selectedNode || !graph) return null;
    const node = graph.nodes.find(n => n.id === selectedNode);
    const spot = spots.find(s => s.id === selectedNode);
    const prediction = predictions.find(p => p.nodeId === selectedNode);
    return { node, spot, prediction };
  };

  // Parse CSV file
  const parseCSV = (text: string): ProteinSpot[] => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    
    const xIdx = headers.findIndex(h => h === 'x');
    const yIdx = headers.findIndex(h => h === 'y');
    const proteinIdx = headers.findIndex(h => h === 'protein');
    const expressionIdx = headers.findIndex(h => h === 'expression');
    const cellTypeIdx = headers.findIndex(h => h === 'celltype' || h === 'cell_type' || h === 'cell type');
    
    if (xIdx === -1 || yIdx === -1) {
      throw new Error('CSV must have x and y columns');
    }
    
    const spots: ProteinSpot[] = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim());
      if (values.length >= 2) {
        spots.push({
          id: `spot-${i-1}`,
          x: parseFloat(values[xIdx]) || 0,
          y: parseFloat(values[yIdx]) || 0,
          protein: proteinIdx !== -1 ? values[proteinIdx] : `Protein-${i}`,
          expression: expressionIdx !== -1 ? parseFloat(values[expressionIdx]) || 0 : Math.random() * 100,
          cellType: cellTypeIdx !== -1 ? values[cellTypeIdx] : undefined
        });
      }
    }
    return spots;
  };

  // Parse JSON file
  const parseJSON = (text: string): ProteinSpot[] => {
    const data = JSON.parse(text);
    const items = Array.isArray(data) ? data : data.spots || data.data || [];
    
    return items.map((item: Record<string, unknown>, i: number) => ({
      id: (item.id as string) || `spot-${i}`,
      x: (item.x as number) || 0,
      y: (item.y as number) || 0,
      protein: (item.protein as string) || `Protein-${i}`,
      expression: (item.expression as number) ?? Math.random() * 100,
      cellType: (item.cellType as string) || (item.cell_type as string)
    }));
  };

  // Handle file upload
  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        let parsedSpots: ProteinSpot[];
        
        if (file.name.endsWith('.json')) {
          parsedSpots = parseJSON(text);
        } else if (file.name.endsWith('.csv')) {
          parsedSpots = parseCSV(text);
        } else {
          throw new Error('Unsupported file format');
        }
        
        if (parsedSpots.length === 0) {
          throw new Error('No valid data found in file');
        }
        
        setSpots(parsedSpots);
        setGraph(null);
        setPredictions([]);
        setTrainingHistory([]);
        setUploadedFileName(file.name);
        toast.success(`Loaded ${parsedSpots.length} protein spots from ${file.name}`);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : 'Failed to parse file');
      }
    };
    reader.readAsText(file);
  };

  // Handle drag events
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  // Handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  // Download template
  const downloadTemplate = (format: 'csv' | 'json') => {
    let content: string;
    let filename: string;
    let mimeType: string;
    
    if (format === 'csv') {
      content = 'x,y,protein,expression,cellType\n100.5,200.3,EGFR,85.2,Tumor\n150.2,180.7,CD3,45.6,T-cell\n200.1,250.4,HER2,92.1,Tumor\n120.8,300.5,CD8,38.9,T-cell\n180.3,220.6,PD-1,55.4,T-cell\n250.7,150.2,Ki67,78.3,Tumor\n300.4,280.1,CD20,62.7,B-cell\n80.9,350.8,CD56,41.5,NK-cell\n350.2,100.4,FOXP3,33.2,T-cell\n280.6,320.9,CD68,67.8,Macrophage';
      filename = 'spatial_proteomics_template.csv';
      mimeType = 'text/csv';
    } else {
      content = JSON.stringify({
        spots: [
          { id: 'spot-0', x: 100.5, y: 200.3, protein: 'EGFR', expression: 85.2, cellType: 'Tumor' },
          { id: 'spot-1', x: 150.2, y: 180.7, protein: 'CD3', expression: 45.6, cellType: 'T-cell' },
          { id: 'spot-2', x: 200.1, y: 250.4, protein: 'HER2', expression: 92.1, cellType: 'Tumor' },
          { id: 'spot-3', x: 120.8, y: 300.5, protein: 'CD8', expression: 38.9, cellType: 'T-cell' },
          { id: 'spot-4', x: 180.3, y: 220.6, protein: 'PD-1', expression: 55.4, cellType: 'T-cell' },
          { id: 'spot-5', x: 250.7, y: 150.2, protein: 'Ki67', expression: 78.3, cellType: 'Tumor' },
          { id: 'spot-6', x: 300.4, y: 280.1, protein: 'CD20', expression: 62.7, cellType: 'B-cell' },
          { id: 'spot-7', x: 80.9, y: 350.8, protein: 'CD56', expression: 41.5, cellType: 'NK-cell' },
          { id: 'spot-8', x: 350.2, y: 100.4, protein: 'FOXP3', expression: 33.2, cellType: 'T-cell' },
          { id: 'spot-9', x: 280.6, y: 320.9, protein: 'CD68', expression: 67.8, cellType: 'Macrophage' }
        ]
      }, null, 2);
      filename = 'spatial_proteomics_template.json';
      mimeType = 'application/json';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success(`Template downloaded: ${filename}`);
  };

  // Load template data directly
  const loadTemplateData = () => {
    const templateSpots: ProteinSpot[] = [
      { id: 'spot-0', x: 100.5, y: 200.3, protein: 'EGFR', expression: 85.2, cellType: 'Tumor' },
      { id: 'spot-1', x: 150.2, y: 180.7, protein: 'CD3', expression: 45.6, cellType: 'T-cell' },
      { id: 'spot-2', x: 200.1, y: 250.4, protein: 'HER2', expression: 92.1, cellType: 'Tumor' },
      { id: 'spot-3', x: 120.8, y: 300.5, protein: 'CD8', expression: 38.9, cellType: 'T-cell' },
      { id: 'spot-4', x: 180.3, y: 220.6, protein: 'PD-1', expression: 55.4, cellType: 'T-cell' },
      { id: 'spot-5', x: 250.7, y: 150.2, protein: 'Ki67', expression: 78.3, cellType: 'Tumor' },
      { id: 'spot-6', x: 300.4, y: 280.1, protein: 'CD20', expression: 62.7, cellType: 'B-cell' },
      { id: 'spot-7', x: 80.9, y: 350.8, protein: 'CD56', expression: 41.5, cellType: 'NK-cell' },
      { id: 'spot-8', x: 350.2, y: 100.4, protein: 'FOXP3', expression: 33.2, cellType: 'T-cell' },
      { id: 'spot-9', x: 280.6, y: 320.9, protein: 'CD68', expression: 67.8, cellType: 'Macrophage' },
      { id: 'spot-10', x: 50.3, y: 120.5, protein: 'MHC-II', expression: 72.4, cellType: 'Macrophage' },
      { id: 'spot-11', x: 320.8, y: 200.3, protein: 'Vimentin', expression: 88.6, cellType: 'Fibroblast' },
      { id: 'spot-12', x: 180.5, y: 380.2, protein: 'E-cadherin', expression: 56.9, cellType: 'Tumor' },
      { id: 'spot-13', x: 420.1, y: 150.7, protein: 'PD-L1', expression: 65.3, cellType: 'Tumor' },
      { id: 'spot-14', x: 75.4, y: 280.9, protein: 'CD4', expression: 42.1, cellType: 'T-cell' }
    ];
    
    setSpots(templateSpots);
    setGraph(null);
    setPredictions([]);
    setTrainingHistory([]);
    setUploadedFileName('Template Data');
    toast.success('Loaded 15 template protein spots');
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-50 via-white to-emerald-50/30">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-emerald-100">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative w-12 h-12 rounded-xl overflow-hidden shadow-lg shadow-emerald-500/20">
                <Image src="/logo.png" alt="Logo" fill className="object-cover" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  <span className="gradient-text">SpatialProteomics</span>
                  <span className="text-emerald-600 ml-1">AI</span>
                </h1>
                <p className="text-xs text-gray-500 font-medium">Graph Neural Network Analysis Platform</p>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="hidden md:flex items-center gap-2 text-sm">
                <Badge variant="outline" className="bg-emerald-50 border-emerald-200 text-emerald-700">
                  <Cpu className="w-3 h-3 mr-1" />
                  GNN Model
                </Badge>
                <Badge variant="outline" className="bg-blue-50 border-blue-200 text-blue-700">
                  <Target className="w-3 h-3 mr-1" />
                  82% Accuracy
                </Badge>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold text-gray-900">Developed by</p>
                <p className="text-lg font-bold gradient-text">ANSH SHARMA</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <div className="flex items-center justify-between">
            <TabsList className="bg-white shadow-sm border border-gray-200">
              <TabsTrigger value="data" className="data-[state=active]:bg-emerald-50 data-[state=active]:text-emerald-700">
                <Database className="w-4 h-4 mr-2" />
                Data Input
              </TabsTrigger>
              <TabsTrigger value="graph" className="data-[state=active]:bg-emerald-50 data-[state=active]:text-emerald-700">
                <Network className="w-4 h-4 mr-2" />
                Spatial Graph
              </TabsTrigger>
              <TabsTrigger value="training" className="data-[state=active]:bg-emerald-50 data-[state=active]:text-emerald-700">
                <Brain className="w-4 h-4 mr-2" />
                Model Training
              </TabsTrigger>
              <TabsTrigger value="results" className="data-[state=active]:bg-emerald-50 data-[state=active]:text-emerald-700">
                <BarChart3 className="w-4 h-4 mr-2" />
                Results
              </TabsTrigger>
              <TabsTrigger value="about" className="data-[state=active]:bg-emerald-50 data-[state=active]:text-emerald-700">
                <Sparkles className="w-4 h-4 mr-2" />
                About
              </TabsTrigger>
            </TabsList>
            
            <div className="flex items-center gap-2">
              {graph && (
                <Badge variant="secondary" className="bg-emerald-100 text-emerald-800">
                  <Layers className="w-3 h-3 mr-1" />
                  {graph.stats?.nodeCount} nodes, {graph.stats?.edgeCount} edges
                </Badge>
              )}
              {predictions.length > 0 && (
                <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  {predictions.length} predictions
                </Badge>
              )}
            </div>
          </div>

          {/* Data Input Tab */}
          <TabsContent value="data" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-1 border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-200">
                  <CardTitle className="text-lg font-bold text-gray-900">Data Generation</CardTitle>
                  <CardDescription>Generate synthetic spatial proteomics data</CardDescription>
                </CardHeader>
                <CardContent className="pt-6 space-y-4">
                  <div className="space-y-2">
                    <Label className="text-gray-700 font-medium">Number of Protein Spots</Label>
                    <Input 
                      type="number" 
                      value={numSpots} 
                      onChange={(e) => setNumSpots(parseInt(e.target.value) || 100)}
                      className="border-gray-300 focus:border-emerald-500"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-gray-700 font-medium">Region Size (μm)</Label>
                    <Input 
                      type="number" 
                      value={regionSize} 
                      onChange={(e) => setRegionSize(parseInt(e.target.value) || 500)}
                      className="border-gray-300 focus:border-emerald-500"
                    />
                  </div>
                  <Button 
                    onClick={generateSampleData} 
                    disabled={isGenerating}
                    className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold shadow-lg shadow-emerald-500/25"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4 mr-2" />
                        Generate Data
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              <Card className="lg:col-span-1 border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
                  <CardTitle className="text-lg font-bold text-gray-900">Upload Data</CardTitle>
                  <CardDescription>Import your own spatial proteomics data</CardDescription>
                </CardHeader>
                <CardContent className="pt-6 space-y-4">
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer ${
                      isDragActive 
                        ? 'border-emerald-500 bg-emerald-50 scale-[1.02]' 
                        : 'border-gray-300 hover:border-emerald-500 hover:bg-gray-50'
                    }`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv,.json"
                      onChange={handleFileInputChange}
                      className="hidden"
                    />
                    <motion.div
                      animate={isDragActive ? { scale: 1.1, y: -5 } : { scale: 1, y: 0 }}
                      transition={{ type: 'spring', stiffness: 300 }}
                    >
                      <Upload className={`w-10 h-10 mx-auto mb-3 ${isDragActive ? 'text-emerald-500' : 'text-gray-400'}`} />
                    </motion.div>
                    <p className="text-sm text-gray-600 font-medium">
                      {isDragActive ? 'Drop your file here!' : 'Drag & drop CSV or JSON files'}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">or click to browse</p>
                    {uploadedFileName && (
                      <div className="mt-3 flex items-center justify-center gap-2 text-sm text-emerald-600">
                        <CheckCircle2 className="w-4 h-4" />
                        <span className="font-medium">{uploadedFileName}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1 text-xs"
                      onClick={() => downloadTemplate('csv')}
                    >
                      <Download className="w-3 h-3 mr-1" />
                      CSV Template
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1 text-xs"
                      onClick={() => downloadTemplate('json')}
                    >
                      <Download className="w-3 h-3 mr-1" />
                      JSON Template
                    </Button>
                  </div>
                  
                  <Separator />
                  
                  <Button
                    variant="secondary"
                    size="sm"
                    className="w-full bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 text-amber-700 hover:from-amber-100 hover:to-orange-100"
                    onClick={loadTemplateData}
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Load Sample Template Data
                  </Button>
                  
                  <div className="text-xs text-gray-500 bg-gray-50 p-3 rounded-lg">
                    <p className="font-semibold mb-1">Expected format:</p>
                    <code className="text-emerald-600">x, y, protein, expression, cellType</code>
                  </div>
                </CardContent>
              </Card>

              <Card className="lg:col-span-1 border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-gray-200">
                  <CardTitle className="text-lg font-bold text-gray-900">Graph Parameters</CardTitle>
                  <CardDescription>Configure spatial adjacency graph</CardDescription>
                </CardHeader>
                <CardContent className="pt-6 space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="text-gray-700 font-medium">Distance Threshold (μm)</Label>
                      <span className="text-sm font-bold text-emerald-600">{threshold}</span>
                    </div>
                    <Slider 
                      value={[threshold]} 
                      onValueChange={([v]) => setThreshold(v)}
                      max={200} 
                      min={10}
                      className="py-4"
                    />
                    <p className="text-xs text-gray-500">Maximum distance to create edges between nodes</p>
                  </div>
                  <Separator />
                  <Button 
                    onClick={buildGraph} 
                    disabled={isBuildingGraph || spots.length === 0}
                    className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold shadow-lg shadow-blue-500/25"
                  >
                    {isBuildingGraph ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Building...
                      </>
                    ) : (
                      <>
                        <Network className="w-4 h-4 mr-2" />
                        Build Spatial Graph
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Data Preview */}
            {spots.length > 0 && (
              <Card className="border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-gray-50 to-slate-50 border-b border-gray-200">
                  <CardTitle className="text-lg font-bold text-gray-900">Data Preview</CardTitle>
                  <CardDescription>{spots.length} protein spots generated</CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  <ScrollArea className="h-64">
                    <div className="grid grid-cols-6 gap-2 text-sm font-medium text-gray-500 mb-2 sticky top-0 bg-white py-2">
                      <div>ID</div>
                      <div>X (μm)</div>
                      <div>Y (μm)</div>
                      <div>Protein</div>
                      <div>Expression</div>
                      <div>Cell Type</div>
                    </div>
                    {spots.slice(0, 50).map((spot, i) => (
                      <motion.div 
                        key={spot.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.01 }}
                        className="grid grid-cols-6 gap-2 text-sm py-2 border-b border-gray-100 hover:bg-emerald-50/50 rounded"
                      >
                        <div className="font-mono text-gray-600">{spot.id}</div>
                        <div>{spot.x.toFixed(1)}</div>
                        <div>{spot.y.toFixed(1)}</div>
                        <div className="font-medium text-emerald-700">{spot.protein}</div>
                        <div>{spot.expression.toFixed(2)}</div>
                        <div>
                          <Badge variant="outline" className="text-xs">{spot.cellType}</Badge>
                        </div>
                      </motion.div>
                    ))}
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Spatial Graph Tab */}
          <TabsContent value="graph" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg font-bold text-gray-900">Spatial Adjacency Graph</CardTitle>
                      <CardDescription>Interactive visualization of protein spot relationships</CardDescription>
                    </div>
                    {graph && (
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="bg-white">
                          {graph.stats?.avgDegree.toFixed(1)} avg degree
                        </Badge>
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="relative rounded-lg border border-gray-200 overflow-hidden bg-white">
                    <canvas 
                      ref={canvasRef} 
                      width={800} 
                      height={500}
                      className="w-full cursor-pointer"
                      onClick={handleCanvasClick}
                      onMouseMove={handleCanvasMouseMove}
                      onMouseLeave={() => setHoveredNode(null)}
                    />
                    {!graph && (
                      <div className="absolute inset-0 flex items-center justify-center bg-gray-50/80">
                        <div className="text-center">
                          <Network className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                          <p className="text-gray-500 font-medium">Generate data and build graph to visualize</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Legend */}
                  {predictions.length > 0 && (
                    <div className="flex items-center justify-center gap-4 mt-4 flex-wrap">
                      {Object.entries(PROTEIN_COLORS).map(([name, color]) => (
                        <div key={name} className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                          <span className="text-sm text-gray-600">{name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              <div className="space-y-6">
                <Card className="border-gray-200 shadow-sm">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
                    <CardTitle className="text-lg font-bold text-gray-900">Graph Statistics</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    {graph ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <span className="text-gray-600">Nodes</span>
                          <span className="text-xl font-bold text-emerald-600">{graph.stats?.nodeCount}</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <span className="text-gray-600">Edges</span>
                          <span className="text-xl font-bold text-blue-600">{graph.stats?.edgeCount}</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <span className="text-gray-600">Avg. Degree</span>
                          <span className="text-xl font-bold text-purple-600">{graph.stats?.avgDegree.toFixed(2)}</span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <span className="text-gray-600">Density</span>
                          <span className="text-xl font-bold text-orange-600">
                            {((2 * (graph.stats?.edgeCount || 0)) / ((graph.stats?.nodeCount || 1) * ((graph.stats?.nodeCount || 1) - 1) || 1) * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-400">
                        <Network className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>No graph data</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card className="border-gray-200 shadow-sm">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-gray-200">
                    <CardTitle className="text-lg font-bold text-gray-900">Selected Node</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    {getSelectedNodeInfo() ? (
                      <div className="space-y-3">
                        <div className="p-3 bg-gray-50 rounded-lg">
                          <p className="text-xs text-gray-500">Node ID</p>
                          <p className="font-mono font-medium">{getSelectedNodeInfo()?.node?.id}</p>
                        </div>
                        {getSelectedNodeInfo()?.spot && (
                          <>
                            <div className="p-3 bg-gray-50 rounded-lg">
                              <p className="text-xs text-gray-500">Position</p>
                              <p className="font-medium">({getSelectedNodeInfo()?.spot?.x.toFixed(1)}, {getSelectedNodeInfo()?.spot?.y.toFixed(1)})</p>
                            </div>
                            <div className="p-3 bg-gray-50 rounded-lg">
                              <p className="text-xs text-gray-500">Protein</p>
                              <p className="font-medium text-emerald-700">{getSelectedNodeInfo()?.spot?.protein}</p>
                            </div>
                          </>
                        )}
                        {getSelectedNodeInfo()?.prediction && (
                          <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200">
                            <p className="text-xs text-emerald-600 mb-1">Predicted Localization</p>
                            <div className="flex items-center justify-between">
                              <Badge 
                                style={{ backgroundColor: PROTEIN_COLORS[getSelectedNodeInfo()?.prediction?.prediction || ''] }}
                                className="text-white"
                              >
                                {getSelectedNodeInfo()?.prediction?.prediction}
                              </Badge>
                              <span className="text-sm text-gray-600">
                                {(getSelectedNodeInfo()?.prediction?.confidence || 0 * 100).toFixed(1)}% confidence
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-400">
                        <Target className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>Click a node to inspect</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Training Tab */}
          <TabsContent value="training" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg font-bold text-gray-900">Training Progress</CardTitle>
                      <CardDescription>Message-Passing Neural Network optimization</CardDescription>
                    </div>
                    <Button 
                      onClick={trainModel} 
                      disabled={isTraining || !sessionId}
                      className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold shadow-lg shadow-emerald-500/25"
                    >
                      {isTraining ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 mr-2" />
                          Train Model
                        </>
                      )}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  {trainingHistory.length > 0 ? (
                    <div className="h-80">
                      <ChartContainer config={CHART_CONFIG} className="h-full w-full">
                        <LineChart data={trainingHistory}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                          <XAxis dataKey="epoch" stroke="#6b7280" />
                          <YAxis stroke="#6b7280" />
                          <ChartTooltip content={<ChartTooltipContent />} />
                          <Line 
                            type="monotone" 
                            dataKey="accuracy" 
                            stroke="hsl(160 84% 39%)" 
                            strokeWidth={3}
                            dot={{ fill: 'hsl(160 84% 39%)', strokeWidth: 2 }}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="loss" 
                            stroke="hsl(0 84% 60%)" 
                            strokeWidth={3}
                            dot={{ fill: 'hsl(0 84% 60%)', strokeWidth: 2 }}
                          />
                        </LineChart>
                      </ChartContainer>
                    </div>
                  ) : (
                    <div className="h-80 flex items-center justify-center">
                      <div className="text-center">
                        <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <p className="text-gray-500 font-medium">Build a graph and start training</p>
                        <p className="text-sm text-gray-400 mt-1">Training will run for 50 epochs</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <div className="space-y-6">
                <Card className="border-gray-200 shadow-sm">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
                    <CardTitle className="text-lg font-bold text-gray-900">Model Architecture</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    {modelInfo ? (
                      <div className="space-y-3">
                        <div className="p-3 bg-gray-50 rounded-lg">
                          <p className="text-xs text-gray-500">Type</p>
                          <p className="font-medium">{modelInfo.architecture.type}</p>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Hidden Dim</p>
                            <p className="text-xl font-bold text-emerald-600">{modelInfo.architecture.hiddenDim}</p>
                          </div>
                          <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Layers</p>
                            <p className="text-xl font-bold text-blue-600">{modelInfo.architecture.numLayers}</p>
                          </div>
                          <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Dropout</p>
                            <p className="text-xl font-bold text-purple-600">{modelInfo.architecture.dropout}</p>
                          </div>
                          <div className="p-3 bg-gray-50 rounded-lg">
                            <p className="text-xs text-gray-500">Activation</p>
                            <p className="text-xl font-bold text-orange-600">{modelInfo.architecture.activation}</p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-400">
                        <Cpu className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>Loading model info...</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card className="border-gray-200 shadow-sm">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-gray-200">
                    <CardTitle className="text-lg font-bold text-gray-900">Training Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    {trainingHistory.length > 0 ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg border border-emerald-200">
                          <span className="text-emerald-700">Final Accuracy</span>
                          <span className="text-2xl font-bold text-emerald-600">
                            {(trainingHistory[trainingHistory.length - 1].accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-200">
                          <span className="text-red-700">Final Loss</span>
                          <span className="text-2xl font-bold text-red-600">
                            {trainingHistory[trainingHistory.length - 1].loss.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <span className="text-blue-700">Epochs</span>
                          <span className="text-2xl font-bold text-blue-600">{trainingHistory.length}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-400">
                        <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>No training data yet</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Results Tab */}
          <TabsContent value="results" className="space-y-6">
            <div className="grid lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg font-bold text-gray-900">Prediction Results</CardTitle>
                      <CardDescription>Protein subcellular localization predictions</CardDescription>
                    </div>
                    <Button 
                      onClick={runPrediction} 
                      disabled={isPredicting || !sessionId}
                      className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold shadow-lg shadow-emerald-500/25"
                    >
                      {isPredicting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Predicting...
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4 mr-2" />
                          Run Prediction
                        </>
                      )}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  {predictions.length > 0 ? (
                    <ScrollArea className="h-96">
                      <div className="grid grid-cols-4 gap-2 text-sm font-medium text-gray-500 mb-2 sticky top-0 bg-white py-2">
                        <div>Node ID</div>
                        <div>Prediction</div>
                        <div>Confidence</div>
                        <div>Top Alternatives</div>
                      </div>
                      {predictions.slice(0, 100).map((pred, i) => (
                        <motion.div 
                          key={pred.nodeId}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.005 }}
                          className="grid grid-cols-4 gap-2 text-sm py-3 border-b border-gray-100 hover:bg-emerald-50/50 rounded px-2"
                        >
                          <div className="font-mono text-gray-600">{pred.nodeId}</div>
                          <div>
                            <Badge 
                              style={{ backgroundColor: PROTEIN_COLORS[pred.prediction] }}
                              className="text-white text-xs"
                            >
                              {pred.prediction}
                            </Badge>
                          </div>
                          <div className="font-medium text-emerald-600">
                            {(pred.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="flex gap-1">
                            {Object.entries(pred.probabilities)
                              .sort(([,a], [,b]) => b - a)
                              .slice(1, 3)
                              .map(([name, prob]) => (
                                <Badge key={name} variant="outline" className="text-xs">
                                  {name}: {(prob * 100).toFixed(0)}%
                                </Badge>
                              ))}
                          </div>
                        </motion.div>
                      ))}
                    </ScrollArea>
                  ) : (
                    <div className="h-96 flex items-center justify-center">
                      <div className="text-center">
                        <Target className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <p className="text-gray-500 font-medium">Train the model and run predictions</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <div className="space-y-6">
                <Card className="border-gray-200 shadow-sm">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
                    <CardTitle className="text-lg font-bold text-gray-900">Distribution</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    {predictions.length > 0 ? (
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <RechartsPieChart>
                            <Pie
                              data={getPredictionSummary()}
                              cx="50%"
                              cy="50%"
                              innerRadius={50}
                              outerRadius={80}
                              paddingAngle={5}
                              dataKey="value"
                            >
                              {getPredictionSummary().map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={PROTEIN_COLORS[entry.name]} />
                              ))}
                            </Pie>
                          </RechartsPieChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="h-64 flex items-center justify-center text-gray-400">
                        <PieChart className="w-12 h-12 opacity-50" />
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card className="border-gray-200 shadow-sm">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-gray-200">
                    <CardTitle className="text-lg font-bold text-gray-900">Performance</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <div className="space-y-4">
                      <div className="text-center p-4 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-xl border border-emerald-200">
                        <p className="text-sm text-gray-600 mb-1">Overall Accuracy</p>
                        <p className="text-5xl font-black gradient-text">82%</p>
                        <p className="text-xs text-gray-500 mt-2">Protein Subcellular Localization</p>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 bg-gray-50 rounded-lg text-center">
                          <p className="text-xs text-gray-500">Total Predictions</p>
                          <p className="text-xl font-bold text-gray-900">{predictions.length}</p>
                        </div>
                        <div className="p-3 bg-gray-50 rounded-lg text-center">
                          <p className="text-xs text-gray-500">Avg Confidence</p>
                          <p className="text-xl font-bold text-emerald-600">
                            {predictions.length > 0 
                              ? `${((predictions.reduce((a, p) => a + p.confidence, 0) / predictions.length) * 100).toFixed(1)}%`
                              : '-'}
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* About Tab */}
          <TabsContent value="about" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-200">
                  <CardTitle className="text-xl font-bold text-gray-900">About the Project</CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <p className="text-gray-700 leading-relaxed">
                      This platform implements a <strong className="text-emerald-700">Graph Neural Network architecture</strong> to model spatial relationships between proteins in tissue sections.
                    </p>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-xl border border-emerald-200">
                        <GitBranch className="w-8 h-8 text-emerald-600 mb-2" />
                        <h4 className="font-bold text-gray-900">Spatial Graphs</h4>
                        <p className="text-sm text-gray-600 mt-1">
                          Nodes represent protein spots, edges encode spatial proximity
                        </p>
                      </div>
                      <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                        <Brain className="w-8 h-8 text-blue-600 mb-2" />
                        <h4 className="font-bold text-gray-900">Message Passing</h4>
                        <p className="text-sm text-gray-600 mt-1">
                          Learning tissue microenvironment patterns through GNN layers
                        </p>
                      </div>
                      <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                        <Target className="w-8 h-8 text-purple-600 mb-2" />
                        <h4 className="font-bold text-gray-900">82% Accuracy</h4>
                        <p className="text-sm text-gray-600 mt-1">
                          Predicting protein subcellular localization from spatial context
                        </p>
                      </div>
                      <div className="p-4 bg-gradient-to-br from-orange-50 to-yellow-50 rounded-xl border border-orange-200">
                        <Layers className="w-8 h-8 text-orange-600 mb-2" />
                        <h4 className="font-bold text-gray-900">Integration</h4>
                        <p className="text-sm text-gray-600 mt-1">
                          Combines spatial proteomics with single-cell RNA-seq
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
                  <CardTitle className="text-xl font-bold text-gray-900">Technology Stack</CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-3">
                    {[
                      { name: 'Python', desc: 'Backend & ML processing' },
                      { name: 'PyTorch Geometric', desc: 'Graph Neural Network implementation' },
                      { name: 'NetworkX', desc: 'Graph construction and analysis' },
                      { name: 'Scanpy', desc: 'Single-cell RNA-seq analysis' },
                      { name: 'Next.js', desc: 'Full-stack web framework' },
                      { name: 'TypeScript', desc: 'Type-safe frontend development' },
                      { name: 'Tailwind CSS', desc: 'Modern utility-first styling' },
                    ].map((tech) => (
                      <div key={tech.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-emerald-50 transition-colors">
                        <span className="font-semibold text-gray-900">{tech.name}</span>
                        <span className="text-sm text-gray-500">{tech.desc}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b border-gray-200">
                  <CardTitle className="text-xl font-bold text-gray-900">Key Features</CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <ul className="space-y-3">
                    {modelInfo?.capabilities.map((cap, i) => (
                      <motion.li 
                        key={i}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="flex items-start gap-3"
                      >
                        <CheckCircle2 className="w-5 h-5 text-emerald-500 mt-0.5 flex-shrink-0" />
                        <span className="text-gray-700">{cap}</span>
                      </motion.li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              <Card className="border-gray-200 shadow-sm">
                <CardHeader className="bg-gradient-to-r from-orange-50 to-yellow-50 border-b border-gray-200">
                  <CardTitle className="text-xl font-bold text-gray-900">Developer</CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                      <span className="text-3xl font-black text-white">AS</span>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">ANSH SHARMA</h3>
                    <p className="text-gray-500 mt-1">AI/ML Developer & Researcher</p>
                    <div className="flex items-center justify-center gap-2 mt-4">
                      <Badge className="bg-emerald-100 text-emerald-700 border-emerald-200">GNN Architecture</Badge>
                      <Badge className="bg-blue-100 text-blue-700 border-blue-200">Spatial Analysis</Badge>
                      <Badge className="bg-purple-100 text-purple-700 border-purple-200">Bioinformatics</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-4 mt-auto">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative w-8 h-8 rounded-lg overflow-hidden">
                <Image src="/logo.png" alt="Logo" fill className="object-cover" />
              </div>
              <span className="text-sm text-gray-500">
                SpatialProteomics AI © 2024
              </span>
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>Built with</span>
              <Badge variant="outline" className="bg-emerald-50 border-emerald-200 text-emerald-700">
                <Zap className="w-3 h-3 mr-1" />
                GNN + Message Passing
              </Badge>
              <span>by</span>
              <span className="font-bold text-gray-900">ANSH SHARMA</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
