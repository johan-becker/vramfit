/**
 * Name lookup for the model and device databases.
 *
 * People type `4090`, `RTX 4090`, `rtx-4090`, `llama3.1:8b` and
 * `Llama 3.1 8B` and expect all of them to work, so every id, name and alias
 * is indexed under one normalised key. The normaliser folds case and collapses
 * spaces and underscores to hyphens, which is exactly the difference between a
 * display name and an id in the bundled data ("Llama 3.1 8B" -> "llama-3.1-8b")
 * -- so names resolve for free rather than needing their own alias entries.
 *
 * Collisions are a hard error at construction time. Two entries answering to
 * the same alias would make lookup depend on file order, and a silently
 * shadowed device is a wrong answer rather than a missing one.
 */

export interface Named {
  id: string;
  name: string;
  aliases: string[];
}

export interface Registry<T extends Named> {
  /** Every entry, in database order. */
  all(): readonly T[];
  /** Resolve an id, name or alias. Undefined when nothing matches. */
  find(query: string): T | undefined;
  /** Resolve or throw, listing near matches in the message. */
  get(query: string): T;
  /** Substring matches over id, name and aliases, for "did you mean". */
  search(query: string): T[];
}

export function normalizeKey(value: string): string {
  return value.trim().toLowerCase().replaceAll(/[\s_]+/g, "-");
}

/** Levenshtein distance, capped: only used to rank a handful of suggestions. */
function editDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++) {
      const substitution = (previous[j - 1] as number) + (a[i - 1] === b[j - 1] ? 0 : 1);
      const deletion = (previous[j] as number) + 1;
      const insertion = (current[j - 1] as number) + 1;
      current.push(Math.min(substitution, deletion, insertion));
    }
    previous = current;
  }
  return previous[b.length] as number;
}

export function createRegistry<T extends Named>(entries: readonly T[], kind: string): Registry<T> {
  const index = new Map<string, T>();

  const claim = (key: string, entry: T, label: string): void => {
    const normalized = normalizeKey(key);
    const existing = index.get(normalized);
    if (existing !== undefined && existing !== entry) {
      throw new Error(
        `Duplicate ${kind} ${label} "${key}": claimed by both "${existing.id}" and "${entry.id}"`,
      );
    }
    index.set(normalized, entry);
  };

  for (const entry of entries) {
    claim(entry.id, entry, "id");
    claim(entry.name, entry, "name");
    for (const alias of entry.aliases) claim(alias, entry, "alias");
  }

  const find = (query: string): T | undefined => index.get(normalizeKey(query));

  const search = (query: string): T[] => {
    const needle = normalizeKey(query);
    if (needle === "") return [...entries];
    return entries.filter((entry) =>
      [entry.id, entry.name, ...entry.aliases].some((candidate) =>
        normalizeKey(candidate).includes(needle),
      ),
    );
  };

  return {
    all: () => entries,
    find,
    search,
    get(query: string): T {
      const found = find(query);
      if (found) return found;

      const needle = normalizeKey(query);
      const substring = search(query).slice(0, 5);
      const suggestions =
        substring.length > 0
          ? substring
          : entries
              .map((entry) => ({ entry, score: editDistance(needle, normalizeKey(entry.id)) }))
              .toSorted((a, b) => a.score - b.score)
              .slice(0, 3)
              .filter((candidate) => candidate.score <= Math.max(3, needle.length / 2))
              .map((candidate) => candidate.entry);

      const hint =
        suggestions.length > 0
          ? ` Did you mean: ${suggestions.map((entry) => entry.id).join(", ")}?`
          : "";
      throw new Error(`Unknown ${kind} "${query}".${hint}`);
    },
  };
}
