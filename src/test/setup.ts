import "@testing-library/jest-dom/vitest";
import * as tf from "@tensorflow/tfjs";

// TF.js parity tests run under jsdom (no WebGL) — pin CPU before any model build.
await tf.setBackend("cpu");
await tf.ready();
