import React from "react";
import { useState, useEffect } from "react";
import useDrawing from "../../hooks/use-drawing-r";

import ImageControl from "./ImageControl";
import BrushSizeSelector from "./BrushSizeSelector";
import Sidebar from "./Sidebar";
import DrawingField from "./DrawingField";

import styles from "./DrawingPanel.module.css";

const DrawingPanel = (props) => {
  const [showDropzone, setShowDropzone] = useState(true);

  const imgSizeMax = 512;

  const [imageSize, setImageSize] = useState({ height: 0, width: 0 });

  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedImgUrl, setSelectedImgUrl] = useState(null);
  const [serverSize, setServerSize] = useState(512);

  const [
    canvasRef,
    brushSize,
    setBrushSize,
    startDrawing,
    finishDrawing,
    draw,
    clearCanvas,
    undoStepHandler,
    redoStepHandler,
    resetBuffers,
  ] = useDrawing();

  const loadImage = (file) => {
    if (!file) {
      return;
    }

    setSelectedFile(file);
    setShowDropzone(false);
    resetBuffers();

    if (selectedImgUrl) {
      URL.revokeObjectURL(selectedImgUrl);
    }
    const imgURL = URL.createObjectURL(file);

    const img = new Image();

    img.onload = () => {
      const minSize = Math.max(img.height, img.width);
      const resizeFactor = imgSizeMax / minSize;
      const newSize = {
        height: Math.round(img.height * resizeFactor),
        width: Math.round(img.width * resizeFactor),
      };
      setImageSize(newSize);
    };
    img.src = imgURL;
    setSelectedImgUrl(imgURL);
  };

  const fileSelectedHandler = (event) => {
    loadImage(event.target.files[0]);
  };

  const dropFileHandler = (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (event.dataTransfer.items) {
      loadImage(event.dataTransfer.items[0].getAsFile());
    } else {
      loadImage(event.dataTransfer.files[0]);
    }
  };

  const fileUploadHandler = () => {
    canvasRef.current.toBlob(async (blob) => {
      const selectedModels = props.modelsData
        .filter((v) => v.is_loaded)
        .map((v) => v.name);

      const fd = new FormData();
      fd.append("image", selectedFile, selectedFile.name);
      fd.append("mask", blob, "mask");
      fd.append("models", selectedModels);
      fd.append("size", serverSize);

      const response = await fetch("/api/inpaint", {
        method: "POST",
        body: fd,
      });

      const data = await response.json();

      if (data) {
        props.onOutput(data.data);
      }
    });
  };

  const outpaintHandler = async ({
    expandRatio,
    padTop,
    padBottom,
    padLeft,
    padRight,
  }) => {
    if (!selectedFile) return;

    const modelsData = Array.isArray(props.modelsData) ? props.modelsData : [];

    const selectedModels = modelsData
      .filter((v) => v.is_loaded)
      .map((v) => v.name);

    if (selectedModels.length === 0) {
      console.warn("No loaded models available for outpaint");
      return;
    }

    const modelName = selectedModels[0]; // use first loaded model for now

    const fd = new FormData();
    fd.append("image", selectedFile, selectedFile.name);
    fd.append("model", modelName);

    // Always send all 5 fields; backend will interpret them
    fd.append("expand_ratio", String(expandRatio ?? 0));
    fd.append("pad_top", String(padTop ?? 0));
    fd.append("pad_bottom", String(padBottom ?? 0));
    fd.append("pad_left", String(padLeft ?? 0));
    fd.append("pad_right", String(padRight ?? 0));

    try {
      const response = await fetch("/api/outpaint", {
        method: "POST",
        body: fd,
      });

      if (!response.ok) {
        console.error(
          "Outpaint request failed:",
          response.status,
          response.statusText
        );
        return;
      }

      const data = await response.json();
      if (data && data.data) {
        props.onOutput(data.data);
      }
    } catch (err) {
      console.error("Outpaint fetch error:", err);
    }
  };

  return (
    <div className={styles["control-panel"]}>
      <div className={styles["wrapper"]}>
        <div className={styles["drawing-container"]}>
          <DrawingField
            imageSize={imageSize}
            imgSizeMax={imgSizeMax}
            canvasRef={canvasRef}
            selectedImgUrl={selectedImgUrl}
            showDropzone={showDropzone}
            setShowDropzone={setShowDropzone}
            dropFileHandler={dropFileHandler}
            startDrawing={startDrawing}
            finishDrawing={finishDrawing}
            draw={draw}
            // lineWidthChange={lineWidthChange}
            setBrushSize={setBrushSize}
          />

          <BrushSizeSelector
            brushSize={brushSize}
            setBrushSize={setBrushSize}
          />
        </div>
        <div className={styles.test}>
          <Sidebar
            serverSize={serverSize}
            onSizeChange={setServerSize}
            canvasRef={canvasRef}
            selectedImgUrl={selectedImgUrl}
          />
        </div>
      </div>

      <ImageControl
        onFileSelected={fileSelectedHandler}
        onInpaintClick={fileUploadHandler}
        onOutpaintClick={outpaintHandler}
        onClearClick={clearCanvas}
        onUndoClick={undoStepHandler}
        onRedoClick={redoStepHandler}
      />
    </div>
  );
};

export default DrawingPanel;
