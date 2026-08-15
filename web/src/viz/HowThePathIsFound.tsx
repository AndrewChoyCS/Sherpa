/**
 * The actual maths behind a route, in the order it runs.
 *
 * Every formula here is transcribed from the source rather than paraphrased —
 * `src/diversity_engine.py` for the distance, `src/curriculum.py` for difficulty,
 * `src/graph.py::edge_cost` for the edge weight, `src/pathfinder.py` for the search.
 * Where a term has a live value for the route currently on screen, that value is shown
 * beside the symbol, so the reader can check the arithmetic against the result rather
 * than take the formula on faith.
 *
 * Numbered 1–4 deliberately. Most numbered lists on a page are decoration, but this one
 * is a genuine pipeline: each stage consumes the previous stage's output, and the order
 * cannot be permuted. Difficulty needs the trajectories, the edge weight needs both
 * difficulty and distance, and the search needs the weighted graph.
 */

import type { GraphPayload, PathPayload } from '../lib/types'
import { num } from '../lib/format'

export interface HowThePathIsFoundProps {
  /** The route on screen, if one has been run — supplies live values. */
  path: PathPayload | null
  graph: GraphPayload | null
}

export function HowThePathIsFound({ path, graph }: HowThePathIsFoundProps) {
  const terms = path?.cost_terms ?? null

  return (
    <ol className="maths">
      {/* ---------------------------------------------------------------- */}
      <li className="maths__step">
        <p className="maths__num">1</p>
        <div className="maths__body">
          <h3 className="maths__title">How far apart are two clips?</h3>
          <p className="prose">
            Every clip is a sequence of end-effector positions, sampled at 30 or 60 fps and
            of a different length from every other clip. Dynamic time warping finds the
            alignment between two sequences that minimises total distance, so a slow fold
            and a fast fold of the same shape score as similar.
          </p>
          <figure className="eq">
            <div className="eq__body">
              <span className="eq__lhs">DTW(a, b)</span>
              <span className="eq__rel">=</span>
              <span className="eq__rhs">
                min<sub>π</sub> √( Σ<sub>(i,j) ∈ π</sub> ‖a<sub>i</sub> − b<sub>j</sub>‖² )
              </span>
            </div>
            <figcaption>
              π is a warping path — a monotonic pairing of the two sequences' indices.
            </figcaption>
          </figure>
          <p className="prose">
            Two corrections matter. The cost is a root-sum-square, so it grows as{' '}
            <code>√T</code> with sequence length; each pair is divided by{' '}
            <code>√(mean length)</code> rather than by the mean, which is the intuitive
            choice and overcorrects. And trajectories are z-scored per axis first, because
            comparing raw extent across a tabletop arm and a head-worn camera measures
            hardware rather than skill.
          </p>
          <figure className="eq">
            <div className="eq__body">
              <span className="eq__lhs">d(a, b)</span>
              <span className="eq__rel">=</span>
              <span className="eq__rhs">
                DTW(a, b) ⁄ √( (T<sub>a</sub> + T<sub>b</sub>) / 2 )
              </span>
            </div>
            <figcaption>Normalised distance, in metres. This is the matrix everything else reads.</figcaption>
          </figure>
        </div>
      </li>

      {/* ---------------------------------------------------------------- */}
      <li className="maths__step">
        <p className="maths__num">2</p>
        <div className="maths__body">
          <h3 className="maths__title">How hard is one clip?</h3>
          <p className="prose">
            Six kinematic features per clip, each robustly standardised, then combined. The
            weights lean on the dimensionless ones on purpose: raw path length varies about
            40× across capture rigs, so weighting it heavily would turn "difficulty" into
            "which rig recorded this".
          </p>
          <figure className="eq">
            <div className="eq__body">
              <span className="eq__lhs">z(x)</span>
              <span className="eq__rel">=</span>
              <span className="eq__rhs">
                ( x − median(x) ) ⁄ ( 1.4826 · MAD(x) )
              </span>
            </div>
            <figcaption>
              Median absolute deviation, not standard deviation — a handful of very long
              human episodes would otherwise set the scale for everything.
            </figcaption>
          </figure>
          <table className="weights">
            <caption className="visually-hidden">Difficulty feature weights</caption>
            <tbody>
              {[
                ['log tortuosity', '1.00', 'path length ÷ straight-line displacement'],
                ['log normalised jerk', '1.00', '√( T⁵/L² · ∫‖jerk‖² dt ) — the standard smoothness measure'],
                ['reversal rate', '0.75', 'direction changes per second'],
                ['path length', '0.50', 'metres travelled'],
                ['workspace span', '0.35', 'bounding extent of the motion'],
                ['duration', '0.25', 'seconds'],
              ].map(([name, weight, note]) => (
                <tr key={name}>
                  <td className="weights__w mono">{weight}</td>
                  <td className="weights__n">{name}</td>
                  <td className="weights__note">{note}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="prose">
            The weighted sum is then mapped to a <strong>percentile rank</strong> in [0, 1]
            rather than min–max rescaled. The raw distribution is heavily right-skewed;
            min–max would compress most clips into a band a few thousandths wide and the
            ramp would be unreadable.
          </p>
        </div>
      </li>

      {/* ---------------------------------------------------------------- */}
      <li className="maths__step">
        <p className="maths__num">3</p>
        <div className="maths__body">
          <h3 className="maths__title">What does one training step cost?</h3>
          <p className="prose">
            This is the whole design. Each clip is a node; a directed edge means "train B
            straight after A". The edge's weight is what makes one ordering better than
            another, and it is the sum of four terms.
          </p>
          <figure className="eq eq--major">
            <div className="eq__body">
              <span className="eq__lhs">w(A → B)</span>
              <span className="eq__rel">=</span>
              <span className="eq__rhs">
                <span className="eq__term" data-term="ramp">ramp</span>
                <span className="eq__op">+</span>
                <span className="eq__term" data-term="interference">interference</span>
                <span className="eq__op">+</span>
                <span className="eq__term" data-term="redundancy">redundancy</span>
                <span className="eq__op">+</span>
                <span className="eq__term" data-term="step">step</span>
              </span>
            </div>
            {terms && (
              <figcaption className="eq__live">
                On the route above, summed over every transition:{' '}
                {Object.entries(terms)
                  .filter(([k]) => k !== 'weight')
                  .map(([k, v]) => `${k} ${num(v, 2)}`)
                  .join('  ·  ')}
                {typeof path?.search_cost === 'number' && (
                  <> &nbsp;=&nbsp; <b>{num(path.search_cost, 2)}</b> total</>
                )}
              </figcaption>
            )}
          </figure>

          <dl className="terms">
            <div data-term="ramp">
              <dt>
                ramp <span className="terms__eq mono">|Δd − τ| / τ&nbsp; (+ 2·|Δd|/τ if Δd &lt; 0)</span>
              </dt>
              <dd>
                Δd is the difficulty change, τ the target increment (0.05). Two-sided on
                purpose: it penalises <em>stalling</em> and <em>leaping</em> equally, and adds
                a surcharge for going backwards. A curriculum that repeats the same
                difficulty forever is as wrong as one that jumps.
              </dd>
            </div>
            <div data-term="interference">
              <dt>
                interference <span className="terms__eq mono">d̂(A,B) + Σ penalties</span>
              </dt>
              <dd>
                The normalised distance, plus a fixed surcharge for each attribute that
                changes across the step: task 0.5, skill family 0.3, embodiment 0.3, capture
                source 0.2. Switching context mid-training is what causes forgetting, so the
                switch has to cost something.
              </dd>
            </div>
            <div data-term="redundancy">
              <dt>
                redundancy <span className="terms__eq mono">max(0, ν − d̂) / ν</span>
              </dt>
              <dd>
                1.0 for an identical clip, falling to 0 at the novelty floor ν (the 5th
                percentile of all distances). Without this term, minimising interference
                means minimising distance — and the cheapest route becomes the same clip
                repeated, a flawless ramp that teaches nothing. It was doing exactly that
                before the term existed.
              </dd>
            </div>
            <div data-term="step">
              <dt>
                step <span className="terms__eq mono">0.1</span>
              </dt>
              <dd>A flat charge per step, so a shorter route wins all else being equal.</dd>
            </div>
          </dl>
          <p className="prose">
            Every term is non-negative — which is not a stylistic choice, it is Dijkstra's
            precondition.
          </p>
        </div>
      </li>

      {/* ---------------------------------------------------------------- */}
      <li className="maths__step">
        <p className="maths__num">4</p>
        <div className="maths__body">
          <h3 className="maths__title">Find the cheapest route, then insert rest stops</h3>
          <p className="prose">
            A virtual <code>START</code> node connects at zero cost to the easiest clips, and
            Dijkstra finds the minimum-weight path from it to the goal clip. Because
            minimising that sum simultaneously minimises ramp roughness, context switching
            and repetition, <strong>the ordering problem and the routing problem are the
            same problem</strong> — there is no second objective to reconcile.
          </p>
          <figure className="eq">
            <div className="eq__body">
              <span className="eq__lhs">route</span>
              <span className="eq__rel">=</span>
              <span className="eq__rhs">
                arg min<sub>p ∈ paths(START → goal)</sub> Σ<sub>(A,B) ∈ p</sub> w(A → B)
              </span>
            </div>
            {graph && (
              <figcaption>
                Searched over {graph.nodes.length} clips and {graph.n_edges} admissible
                transitions{path ? `, returning ${path.route.length} stages` : ''}.
              </figcaption>
            )}
          </figure>
          <p className="prose">
            Then every fourth stage, the route revisits one clip it has already taught: the
            skill family absent longest, and within it the clip furthest from where training
            currently is. Those are the camps on the ridge. This mirrors experience replay,
            and it is an <em>unvalidated proxy</em> for preventing forgetting — measuring it
            properly would need the training run this project does not include.
          </p>
        </div>
      </li>
    </ol>
  )
}
