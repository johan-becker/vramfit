import { closeSync, fstatSync, openSync, readSync } from "node:fs";
import { readGgufHeader, type ByteSource, type GgufHeader, type GgufReadOptions } from "./reader.js";

/**
 * The file-backed `ByteSource`.
 *
 * Deliberately separate from `reader.ts`, which stays pure: the parser is the
 * part worth testing byte by byte, and it should be usable over anything --
 * a fixture in memory, a slice of an archive -- rather than only over a path.
 * This module is the only place in the GGUF reader that touches the disk, and
 * it opens the file read-only, reads a few hundred kilobytes and closes it.
 */

export interface FileByteSource extends ByteSource {
  close(): void;
}

/** Open a file for random access. The caller closes it. */
export function openByteSource(path: string): FileByteSource {
  const fd = openSync(path, "r");
  let size: number;
  try {
    size = fstatSync(fd).size;
  } catch (cause) {
    closeSync(fd);
    throw cause;
  }

  return {
    size,
    read(offset: number, length: number): Uint8Array {
      const wanted = Math.max(0, Math.min(length, size - offset));
      const buffer = Buffer.allocUnsafe(wanted);
      let filled = 0;
      // readSync is allowed to return a short read; loop until it stops
      // making progress rather than assuming one call is enough.
      while (filled < wanted) {
        const read = readSync(fd, buffer, filled, wanted - filled, offset + filled);
        if (read <= 0) break;
        filled += read;
      }
      return new Uint8Array(buffer.buffer, buffer.byteOffset, filled);
    },
    close(): void {
      closeSync(fd);
    },
  };
}

/** Read one GGUF file's header, opening and closing the file around it. */
export function readGgufFile(path: string, options: GgufReadOptions = {}): GgufHeader {
  const source = openByteSource(path);
  try {
    return readGgufHeader(source, options);
  } finally {
    source.close();
  }
}
