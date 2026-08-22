import type { Io } from "../src/cli/index.js";
import type { ClosableByteSource, PathKind } from "../src/cli/source.js";

/**
 * The CLI's world, in memory.
 *
 * `run(argv, io)` is a pure function of its arguments and this object, so a
 * command that reads a 5 GB checkpoint can be tested without one existing:
 * text files, binary files and directories are all just entries in a Map.
 * `closed` records that every file handle was released, including on the
 * paths that throw.
 */
export interface FakeIoContents {
  text?: Record<string, string>;
  binary?: Record<string, Uint8Array>;
  directories?: readonly string[];
  /** Whether the command should believe it is writing to a terminal. */
  isTty?: boolean;
  /** The environment the colour rules read. */
  env?: Record<string, string>;
}

export class FakeIo implements Io {
  readonly stdout: string[] = [];
  readonly stderr: string[] = [];
  readonly closed: string[] = [];
  private readonly text: Map<string, string>;
  private readonly binary: Map<string, Uint8Array>;
  private readonly directories: Set<string>;
  private readonly environment: Record<string, string>;
  readonly isTty: boolean;

  constructor(contents: FakeIoContents = {}) {
    this.text = new Map(Object.entries(contents.text ?? {}));
    this.binary = new Map(Object.entries(contents.binary ?? {}));
    this.directories = new Set(contents.directories ?? []);
    this.environment = contents.env ?? {};
    this.isTty = contents.isTty ?? false;
  }

  env(name: string): string | undefined {
    return this.environment[name];
  }

  out(text: string): void {
    this.stdout.push(text);
  }

  err(text: string): void {
    this.stderr.push(text);
  }

  readFile(path: string): string {
    const content = this.text.get(path);
    if (content === undefined) throw new Error(`ENOENT: ${path}`);
    return content;
  }

  openBytes(path: string): ClosableByteSource {
    const bytes = this.binary.get(path);
    if (bytes === undefined) throw new Error(`ENOENT: ${path}`);
    const closed = this.closed;
    return {
      size: bytes.length,
      read(offset: number, length: number): Uint8Array {
        return bytes.subarray(offset, offset + Math.max(0, length));
      },
      close(): void {
        closed.push(path);
      },
    };
  }

  pathKind(path: string): PathKind {
    if (this.directories.has(path)) return "directory";
    if (this.binary.has(path) || this.text.has(path)) return "file";
    return "missing";
  }

  get output(): string {
    return this.stdout.join("\n");
  }

  get errors(): string {
    return this.stderr.join("\n");
  }
}
