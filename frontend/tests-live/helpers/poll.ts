type Predicate = () => boolean | Promise<boolean>

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/**
 * True if predicate becomes true within windowMs (real clock). The predicate is sampled every
 * intervalMs, so a state that lasts less than intervalMs may be missed.
 */
export async function observesWithin(predicate: Predicate, windowMs: number, intervalMs = 100): Promise<boolean> {
  const end = Date.now() + windowMs
  do {
    if (await predicate()) return true
    await sleep(intervalMs)
  } while (Date.now() < end)
  return false
}

/** True if predicate never holds during the window. Same sampling limit: min observable duration = intervalMs. */
export async function neverObserves(predicate: Predicate, windowMs: number, intervalMs = 100): Promise<boolean> {
  return !(await observesWithin(predicate, windowMs, intervalMs))
}
