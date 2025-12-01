import React, { useState } from "react";
import styles from "./ImageControl.module.css";
import DoubleButton from "../UI/DoubleButton";

const ImageControl = (props) => {
  const {
    onFileSelected,
    onClearClick,
    onUndoClick,
    onRedoClick,
    onInpaintClick,
    onOutpaintClick,
  } = props;

  // Outpaint parameters (UI state)
  const [expandRatio, setExpandRatio] = useState(0.25);
  const [padTop, setPadTop] = useState(0);
  const [padBottom, setPadBottom] = useState(0);
  const [padLeft, setPadLeft] = useState(0);
  const [padRight, setPadRight] = useState(0);

  const handleOutpaintClick = () => {
    if (!onOutpaintClick) return;

    onOutpaintClick({
      expandRatio,
      padTop,
      padBottom,
      padLeft,
      padRight,
    });
  };

  return (
    <div className={styles["image-control"]}>
      <button className="button-upload">
        <label
          htmlFor={styles["image-upload"]}
          className={styles["image-upload-label"]}
        >
          Choose File
        </label>
        <input
          id={styles["image-upload"]}
          type="file"
          onChange={onFileSelected}
        />
      </button>

      <button className="button-clear" onClick={onClearClick}>
        Clear
      </button>

      <DoubleButton
        labelLeft="Undo"
        labelRight="Redo"
        onClickLeft={onUndoClick}
        onClickRight={onRedoClick}
      />

      {/* Outpaint controls */}
      <div className={styles["outpaint-controls"]}>
        <div className={styles["outpaint-row"]}>
          <label>
            Expand ratio
            <input
              type="number"
              step="0.05"
              min="0"
              max="0.7"
              value={expandRatio}
              onChange={(e) =>
                setExpandRatio(parseFloat(e.target.value) || 0)
              }
            />
          </label>
        </div>

        <div className={styles["outpaint-row"]}>
          <label>
            Pad top
            <input
              type="number"
              min="0"
              value={padTop}
              onChange={(e) => setPadTop(parseInt(e.target.value || "0", 10))}
            />
          </label>
          <label>
            Pad bottom
            <input
              type="number"
              min="0"
              value={padBottom}
              onChange={(e) =>
                setPadBottom(parseInt(e.target.value || "0", 10))
              }
            />
          </label>
        </div>

        <div className={styles["outpaint-row"]}>
          <label>
            Pad left
            <input
              type="number"
              min="0"
              value={padLeft}
              onChange={(e) =>
                setPadLeft(parseInt(e.target.value || "0", 10))
              }
            />
          </label>
          <label>
            Pad right
            <input
              type="number"
              min="0"
              value={padRight}
              onChange={(e) =>
                setPadRight(parseInt(e.target.value || "0", 10))
              }
            />
          </label>
        </div>
      </div>

      {/* Inpaint button */}
      <button
        className={styles["button-inpaint"]}
        onClick={onInpaintClick}
      >
        Inpaint
      </button>

      {/* Outpaint button */}
      <button
        className={styles["button-outpaint"] || styles["button-inpaint"]}
        onClick={handleOutpaintClick}
      >
        Outpaint
      </button>
    </div>
  );
};

export default ImageControl;
