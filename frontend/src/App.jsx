import React, { useMemo, useState } from 'react';
import { preserveTaskCompletion, taskIdentity } from './taskChecklist.js';
import {
  ArrowDown, ArrowRight, CalendarDays, Check, CheckCircle2, Circle, Clock3,
  IndianRupee, Leaf, MapPin, PartyPopper, Sparkles, Users, WandSparkles,
} from 'lucide-react';

const initial = {
  event_type: 'Birthday',
  description: '',
  event_date: '',
  budget_inr: 100000,
  guest_count: 40,
  venue_preference: 'Flexible',
  food_preferences: [],
  dietary_preferences: [],
  cultural_preferences: '',
  accessibility_preferences: '',
};

const foods = ['Vegetarian', 'Vegan', 'Halal', 'Jain'];
const diets = ['Gluten-free', 'No nuts', 'Dairy-free', 'Low spice'];
const money = (value) => `₹${Number(value || 0).toLocaleString('en-IN')}`;
const dateLabel = (value) => new Date(`${value}T12:00:00`).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });

function App() {
  const [form, setForm] = useState(initial);
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [done, setDone] = useState({});
  const [adjustGuests, setAdjustGuests] = useState('');
  const [adjustBudget, setAdjustBudget] = useState('');
  const [mobileFormOpen, setMobileFormOpen] = useState(true);
  const [shockType, setShockType] = useState('category_percentage');
  const [shockCategory, setShockCategory] = useState('');
  const [shockPercent, setShockPercent] = useState(20);
  const [shockAdditionalGuests, setShockAdditionalGuests] = useState(10);
  const [shockExpenseLabel, setShockExpenseLabel] = useState('');
  const [shockExpenseAmount, setShockExpenseAmount] = useState('');
  const [shockResult, setShockResult] = useState(null);
  const [shockLoading, setShockLoading] = useState(false);
  const [shockError, setShockError] = useState('');
  const [recoveryResult, setRecoveryResult] = useState(null);
  const [recoveryLoading, setRecoveryLoading] = useState(false);
  const [recoveryError, setRecoveryError] = useState('');
  const [recoveryApplied, setRecoveryApplied] = useState(false);
  const [tradeOffResult, setTradeOffResult] = useState(null);
  const [tradeOffLoading, setTradeOffLoading] = useState(false);
  const [tradeOffError, setTradeOffError] = useState('');

  const progress = useMemo(() => {
    if (!plan) return 0;
    const completedCount = plan.timeline.filter((task) => done[taskIdentity(task)]).length;
    return plan.timeline.length ? Math.round(completedCount / plan.timeline.length * 100) : 0;
  }, [done, plan]);

  function update(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function toggleList(key, item) {
    update(key, form[key].includes(item) ? form[key].filter((entry) => entry !== item) : [...form[key], item]);
  }

  async function generate(event, overrides = {}) {
    event?.preventDefault();
    setLoading(true);
    setError('');
    try {
      const request = { ...form, ...overrides };
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/api/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = body.detail;
        throw new Error(typeof detail === 'string' ? detail : 'Please check your event details and try again.');
      }
      setPlan(body);
      setDone((current) => preserveTaskCompletion(plan?.timeline || [], current, body.timeline));
      setShockResult(null);
      setShockError('');
      setRecoveryResult(null);
      setRecoveryError('');
      setRecoveryApplied(false);
      setTradeOffResult(null);
      setTradeOffError('');
      setShockCategory(body.budget.allocations[0]?.category || '');
      setMobileFormOpen(false);
      document.getElementById('plan-results')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      setError(err.message || 'Could not reach EventEase. Make sure the backend is running and try again.');
    } finally {
      setLoading(false);
    }
  }

  function regenerateWhatIf(event) {
    const overrides = {};
    if (adjustGuests) overrides.guest_count = Number(adjustGuests);
    if (adjustBudget) overrides.budget_inr = Number(adjustBudget);
    if (Object.keys(overrides).length) {
      setForm((current) => ({ ...current, ...overrides }));
      generate(event, overrides);
    } else {
      generate(event);
    }
  }

  async function simulateBudgetShock(event) {
    event.preventDefault();
    setShockLoading(true);
    setShockError('');
    setShockResult(null);
    const request = {
      baseline_budget_inr: plan.budget.budget_inr,
      baseline_plan_cost_inr: plan.budget.total_estimated_inr,
      baseline_guest_count: plan.event.guest_count,
      baseline_allocations: plan.budget.allocations.map(({ category, amount_inr, percentage, rationale }) => ({
        category, amount_inr, percentage, rationale,
      })),
      shock_type: shockType,
    };
    if (shockType === 'category_percentage') {
      request.category = shockCategory;
      request.increase_percent = Number(shockPercent);
    } else if (shockType === 'additional_guests') {
      request.additional_guests = Number(shockAdditionalGuests);
    } else {
      request.expense_label = shockExpenseLabel;
      request.expense_amount_inr = Number(shockExpenseAmount);
    }
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/api/budget-shock`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = body.detail;
        throw new Error(typeof detail === 'string' ? detail : 'Check the shock details and try again.');
      }
      setShockResult(body);
    } catch (err) {
      setShockError(err.message || 'Could not run the budget simulation. Please try again.');
    } finally {
      setShockLoading(false);
    }
  }

  async function analyzeTimelineRecovery() {
    setRecoveryLoading(true);
    setRecoveryError('');
    setRecoveryResult(null);
    setRecoveryApplied(false);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/api/timeline-recovery`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_date: plan.event.date,
          tasks: plan.timeline.map(({ title, due_date, detail }) => ({ title, due_date, detail })),
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = body.detail;
        const validationMessage = Array.isArray(detail)
          ? detail.map((item) => item.msg).filter(Boolean).join(' ')
          : '';
        throw new Error(typeof detail === 'string' ? detail : validationMessage || 'Check the event timeline and try again.');
      }
      setRecoveryResult(body);
    } catch (err) {
      setRecoveryError(err.message || 'Could not analyze the timeline. Please try again.');
    } finally {
      setRecoveryLoading(false);
    }
  }

  function applyTimelineRecovery() {
    if (!recoveryResult?.can_apply) return;
    setPlan((current) => current && ({
      ...current,
      timeline: current.timeline.map((task, index) => ({
        ...task,
        due_date: recoveryResult.tasks[index].suggested_due_date,
      })),
    }));
    setRecoveryResult(null);
    setRecoveryApplied(true);
  }

  async function compareTradeOffs() {
    setTradeOffLoading(true);
    setTradeOffError('');
    setTradeOffResult(null);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/api/trade-offs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          budget_inr: plan.budget.budget_inr,
          guest_count: plan.event.guest_count,
          baseline_total_inr: plan.budget.total_estimated_inr,
          allocations: plan.budget.allocations.map(({ category, amount_inr }) => ({ category, amount_inr })),
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        const detail = body.detail;
        const validationMessage = Array.isArray(detail)
          ? detail.map((item) => item.msg).filter(Boolean).join(' ')
          : '';
        throw new Error(typeof detail === 'string' ? detail : validationMessage || 'Check the current plan and try again.');
      }
      setTradeOffResult(body);
    } catch (err) {
      setTradeOffError(err.message || 'Could not compare the planning options. Please try again.');
    } finally {
      setTradeOffLoading(false);
    }
  }

  const minDate = new Date().toISOString().slice(0, 10);
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="EventEase home">
          <span className="brand-mark"><PartyPopper size={19} /></span>
          <span>eventease<span className="brand-ai">.ai</span></span>
        </a>
        <div className="topbar-right">
          <span className="topbar-note"><span className="status-dot" /> Your celebration, in good hands</span>
          <button className="top-cta" onClick={() => document.getElementById('planner')?.scrollIntoView({ behavior: 'smooth' })}>
            Start planning <ArrowRight size={15} />
          </button>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><Sparkles size={14} /> PROMPTWARS VIRTUAL 2026 · CHALLENGE 05</div>
            <h1>Make room for<br /><span>the good stuff.</span></h1>
            <p className="hero-sub">A little less spreadsheet. A lot more celebration.<br className="desktop-break" /> Plan something wonderful, without losing sight of your budget.</p>
            <div className="hero-points">
              <span><Check size={15} /> Budget-first planning</span>
              <span><Check size={15} /> Made for your people</span>
            </div>
            <button className="hero-button" onClick={() => document.getElementById('planner')?.scrollIntoView({ behavior: 'smooth' })}>
              Plan your celebration <ArrowDown size={16} />
            </button>
          </div>
          <div className="hero-art" aria-label="Illustration of a celebration table" role="img">
            <div className="art-sun" />
            <div className="art-confetti confetti-one">✳</div><div className="art-confetti confetti-two">✦</div>
            <div className="art-card">
              <div className="art-ribbon">A DAY TO REMEMBER</div>
              <div className="art-cake"><span /><i /><b /></div>
              <div className="art-flowers">✿ <span>✿</span> ✿</div>
              <div className="art-caption">Good things<br />are worth celebrating.</div>
              <div className="art-date">TOGETHER IS THE BEST PLACE TO BE</div>
            </div>
            <div className="art-sticker">made<br />with ♡</div>
          </div>
        </section>

        <section className="planner-section" id="planner">
          <div className="section-intro">
            <div><span className="section-kicker">YOUR PLANNING DESK</span><h2>Let’s make it yours.</h2></div>
            <p>Tell us a little about your day.<br />We’ll take it from there.</p>
          </div>
          <div className="planner-layout">
            <button className="mobile-form-toggle" onClick={() => setMobileFormOpen(!mobileFormOpen)}>
              {mobileFormOpen ? 'Close event details' : 'Edit event details'} <ArrowDown size={15} />
            </button>
            <form className={`planner-card ${mobileFormOpen ? 'mobile-open' : ''}`} onSubmit={generate} id="planner-form">
              <div className="form-heading"><div className="form-icon"><PartyPopper size={18} /></div><div><h3>The celebration</h3><p>The details that make it special.</p></div><span className="step-count">01 / 01</span></div>
              <div className="field-row">
                <label className="field">Event type
                  <select value={form.event_type} onChange={(e) => update('event_type', e.target.value)}>
                    {['Birthday', 'Wedding', 'Corporate event', 'Community gathering', 'Anniversary', 'Other'].map((x) => <option key={x}>{x}</option>)}
                  </select>
                </label>
                <label className="field">Date
                  <input required type="date" min={minDate} value={form.event_date} onChange={(e) => update('event_date', e.target.value)} />
                </label>
              </div>
              <label className="field">What are we celebrating?
                <textarea required minLength={8} maxLength={500} placeholder="A sun-drenched 30th birthday for my favourite people…" value={form.description} onChange={(e) => update('description', e.target.value)} rows="2" />
                <span className="field-hint">{form.description.length}/500 characters</span>
              </label>
              <div className="field-row">
                <label className="field">Your budget <span className="input-affix"><IndianRupee size={14} /> INR</span>
                  <input required type="number" min="1000" max="10000000" step="1000" value={form.budget_inr} onChange={(e) => update('budget_inr', Number(e.target.value))} />
                </label>
                <label className="field">Guest count <span className="input-affix"><Users size={14} /> people</span>
                  <input required type="number" min="1" max="10000" value={form.guest_count} onChange={(e) => update('guest_count', Number(e.target.value))} />
                </label>
              </div>
              <label className="field">Venue feel
                <select value={form.venue_preference} onChange={(e) => update('venue_preference', e.target.value)}>
                  {['Flexible', 'At home', 'Outdoor', 'Banquet hall', 'Community space'].map((x) => <option key={x}>{x}</option>)}
                </select>
              </label>
              <fieldset className="choice-group"><legend>On the menu <span>Choose any</span></legend><div className="chips">{foods.map((x) => <button type="button" key={x} className={`chip ${form.food_preferences.includes(x) ? 'selected' : ''}`} onClick={() => toggleList('food_preferences', x)}>{form.food_preferences.includes(x) && <Check size={12} />}{x}</button>)}</div></fieldset>
              <fieldset className="choice-group"><legend>Dietary needs <span>Choose any</span></legend><div className="chips">{diets.map((x) => <button type="button" key={x} className={`chip ${form.dietary_preferences.includes(x) ? 'selected' : ''}`} onClick={() => toggleList('dietary_preferences', x)}>{form.dietary_preferences.includes(x) && <Check size={12} />}{x}</button>)}</div></fieldset>
              <label className="field">Cultural touches <span className="optional">OPTIONAL</span>
                <input maxLength="300" placeholder="A tradition, ritual, or little detail to include" value={form.cultural_preferences} onChange={(e) => update('cultural_preferences', e.target.value)} />
              </label>
              <label className="field">Accessibility needs <span className="optional">OPTIONAL</span>
                <input maxLength="300" placeholder="Step-free access, quiet space, seating…" value={form.accessibility_preferences} onChange={(e) => update('accessibility_preferences', e.target.value)} />
              </label>
              <button className="generate-button" type="submit" disabled={loading}>
                {loading ? <><span className="spinner" /> Dreaming up your plan…</> : <><WandSparkles size={17} /> Create my celebration plan <ArrowRight size={16} /></>}
              </button>
              {error && <div className="error-message" role="alert">{error}</div>}
              <p className="privacy-note"><CheckCircle2 size={14} /> Your details stay in this planning session.</p>
            </form>

            <section className={`results-column ${plan ? 'has-plan' : ''}`} id="plan-results" aria-live="polite">
              {!plan ? <div className="empty-state">
                <div className="empty-illustration"><span>✳</span><PartyPopper size={43} /><i>✦</i></div>
                <span className="section-kicker">A LITTLE SOMETHING TO LOOK FORWARD TO</span>
                <h3>Your day starts here.</h3>
                <p>Share a few details and your celebration plan will appear right here — thoughtfully tailored, budget checked, and ready to make your own.</p>
                <div className="empty-perks"><span><IndianRupee size={15} /> Every rupee accounted for</span><span><Clock3 size={15} /> A timeline you can follow</span><span><Leaf size={15} /> Your preferences, remembered</span></div>
              </div> : <>
                <div className="results-header">
                  <div><span className="section-kicker">{plan.mode === 'ai' ? 'YOUR AI-GENERATED PLAN' : 'YOUR SAMPLE PLAN'}</span><h2>{plan.event.type}, thoughtfully planned.</h2></div>
                  <span className={`mode-pill ${plan.mode}`}>{plan.mode === 'ai' ? <><Sparkles size={12} /> AI PLAN</> : <>DEMO MODE</>}</span>
                </div>
                {plan.mode === 'demo' && <div className="demo-note">{plan.fallback_reason || 'This is a deterministic sample plan — not generated by AI. Add a Gemini API key to enable personalized AI plans.'}</div>}
                <article className="overview-card">
                  <div className="overview-top"><span className="overview-icon"><Sparkles size={17} /></span><span className="section-kicker">YOUR DAY, AT A GLANCE</span></div>
                  <p>{plan.overview}</p>
                  <div className="overview-meta"><span><CalendarDays size={14} /> {dateLabel(plan.event.date)}</span><span><Users size={14} /> {Number(plan.event.guest_count).toLocaleString('en-IN')} guests</span><span><MapPin size={14} /> {plan.event.venue_preference}</span></div>
                </article>
                <article className={`budget-card ${plan.budget.over_budget ? 'over-budget' : ''}`}>
                  <div className="card-title-row"><div><span className="section-kicker">BUDGETGUARD</span><h3>Every rupee, accounted for.</h3></div><span className="guard-icon"><IndianRupee size={18} /></span></div>
                  <div className="budget-numbers">
                    <div><span>Estimated total</span><strong>{money(plan.budget.total_estimated_inr)}</strong></div>
                    <div><span>Your budget</span><strong>{money(plan.budget.budget_inr)}</strong></div>
                    <div><span>Per guest</span><strong>{money(plan.budget.cost_per_guest_inr)}</strong></div>
                  </div>
                  <div className="budget-track" role="img" aria-label={`Estimated cost is ${Math.round(plan.budget.total_estimated_inr / plan.budget.budget_inr * 100)} percent of budget`}>
                    <span style={{ width: `${Math.min(plan.budget.total_estimated_inr / plan.budget.budget_inr * 100, 100)}%` }} />
                  </div>
                  {plan.budget.over_budget ? <div className="budget-warning" role="alert"><strong>A little over budget</strong><span>By {money(Math.abs(plan.budget.remaining_inr))} — try these thoughtful trims:</span></div> : <div className="budget-ok"><CheckCircle2 size={15} /> {money(plan.budget.remaining_inr)} left to enjoy (or keep as a buffer)</div>}
                  <div className="allocation-list">{plan.budget.allocations.map((item) => <div className="allocation" key={item.category}>
                    <div className="allocation-top"><span>{item.category}</span><strong>{money(item.amount_inr)}</strong></div>
                    <div className="allocation-bar"><span style={{ width: `${Math.min(item.percentage, 100)}%` }} /></div>
                    <div className="allocation-foot"><small>{item.rationale}</small><span>{item.percentage}%</span></div>
                  </div>)}</div>
                  <div className="savings-box"><div><Sparkles size={15} /><strong>{plan.budget.over_budget ? 'Budget-friendly ideas' : 'A little planning magic'}</strong></div><ul>{plan.budget.savings_suggestions.map((item) => <li key={item}>{item}</li>)}</ul></div>
                  <p className="estimate-note">All figures are planning estimates, not confirmed vendor quotations.</p>
                </article>
                <article className="trade-off-card" aria-labelledby="trade-off-title">
                  <div className="card-title-row">
                    <div><span className="section-kicker">COMPARE YOUR OPTIONS</span><h3 id="trade-off-title">AI Trade-Off Explorer</h3></div>
                    <span className="guard-icon"><Sparkles size={17} /></span>
                  </div>
                  <p className="trade-off-intro">A deterministic planning comparison based on your current plan—not an AI-generated recommendation or live vendor-price search.</p>
                  <button className="generate-button trade-off-button" type="button" disabled={tradeOffLoading} onClick={compareTradeOffs}>
                    {tradeOffLoading ? <><span className="spinner" /> Comparing options…</> : <>Compare three options <ArrowRight size={15} /></>}
                  </button>
                  {tradeOffError && <div className="error-message" role="alert">{tradeOffError}</div>}
                  {tradeOffResult && <div className="trade-off-results" aria-live="polite">
                    <div className="trade-off-options">
                      {tradeOffResult.options.map((option) => {
                        const balanced = tradeOffResult.options.find((item) => item.name === 'Balanced');
                        const delta = option.estimated_total_inr - balanced.estimated_total_inr;
                        return <section className={`trade-off-option trade-off-${option.name.toLowerCase().replace(' ', '-')}`} key={option.name}>
                          <h4>{option.name}</h4>
                          <div className="trade-off-metrics">
                            <div><span>Estimated total</span><strong>{money(option.estimated_total_inr)}</strong></div>
                            <div><span>Cost per guest</span><strong>{money(option.estimated_cost_per_guest_inr)}</strong></div>
                            <div><span>Remaining budget</span><strong>{money(option.remaining_budget_inr)}</strong></div>
                            <div><span>{delta <= 0 ? 'Savings vs Balanced' : 'Additional spend vs Balanced'}</span><strong>{money(Math.abs(delta))}</strong></div>
                          </div>
                          {option.exceeds_budget && <div className="trade-off-over-budget" role="alert">This option exceeds the event budget.</div>}
                          {!option.exceeds_budget && <div className="trade-off-within-budget" role="status">Within budget</div>}
                          <p>{option.explanation}</p>
                          <h5>Category comparison</h5>
                          <ul>{option.allocations.map((allocation) => <li key={allocation.category}>
                            <span>{allocation.category}: {money(allocation.baseline_amount_inr)} → {money(allocation.proposed_amount_inr)}</span>
                            <strong>{allocation.change_inr > 0 ? '+' : ''}{money(allocation.change_inr)}</strong>
                          </li>)}</ul>
                          <small className="trade-off-estimate">{option.estimate_notice}</small>
                        </section>;
                      })}
                    </div>
                    <p className="estimate-note">{tradeOffResult.notice}</p>
                  </div>}
                </article>
                <article className="timeline-card">
                  <div className="card-title-row"><div><span className="section-kicker">THE COUNTDOWN</span><h3>Your feel-good checklist.</h3></div><span className="progress-text">{progress}% done</span></div>
                  <div className="progress-bar"><span style={{ width: `${progress}%` }} /></div>
                  <div className="timeline-list">{plan.timeline.map((task, index) => {
                    const taskKey = taskIdentity(task);
                    const isDone = Boolean(done[taskKey]);
                    return <button className={`timeline-item ${isDone ? 'completed' : ''}`} key={taskKey} onClick={() => setDone((current) => ({ ...current, [taskKey]: !current[taskKey] }))}>
                      <span className="task-check">{isDone ? <CheckCircle2 size={20} /> : <Circle size={20} />}</span>
                      <span className="task-content"><strong>{task.title}</strong><span>{task.detail}</span><small><CalendarDays size={12} /> By {dateLabel(task.due_date)}</small></span>
                      <span className="task-number">0{index + 1}</span>
                    </button>;
                  })}</div>
                  <section className="adaptive-recovery" aria-labelledby="adaptive-recovery-title">
                    <div className="adaptive-recovery-heading">
                      <div><span className="section-kicker">SCHEDULE SUPPORT</span><h4 id="adaptive-recovery-title">Adaptive Recovery Planner</h4></div>
                      <button className="recovery-analyze-button" type="button" disabled={recoveryLoading} onClick={analyzeTimelineRecovery}>
                        {recoveryLoading ? <><span className="spinner" /> Analyzing…</> : 'Analyze timeline'}
                      </button>
                    </div>
                    <p className="recovery-helper">Review overdue or out-of-window deadlines. Analysis does not change your event date or checklist.</p>
                    {recoveryError && <div className="error-message" role="alert">{recoveryError}</div>}
                    {recoveryApplied && <div className="recovery-applied" role="status">Suggested dates applied to the checklist. Your event date and task completion remain unchanged.</div>}
                    {recoveryResult && <div className="recovery-analysis" aria-live="polite">
                      <p className="recovery-window">Planning window: {dateLabel(recoveryResult.window_start)}{recoveryResult.window_end && recoveryResult.window_end !== recoveryResult.window_start ? ` – ${dateLabel(recoveryResult.window_end)}` : ''}</p>
                      {recoveryResult.has_unresolved_conflicts && <div className="recovery-conflict" role="alert">Some conflicts cannot be resolved within the available dates. No event date or task deadline has been changed.</div>}
                      {!recoveryResult.has_unresolved_conflicts && <p className="recovery-clear" role="status">All tasks have dates in the available planning window.</p>}
                      <div className="recovery-task-list">{recoveryResult.tasks.map((task, index) => (
                        <div className={`recovery-task recovery-task-${task.status}`} key={`${task.title}-${index}`}>
                          <div className="recovery-task-title"><strong>{task.title}</strong><span className={`recovery-status status-${task.status}`}>{task.status.replaceAll('_', ' ')}</span></div>
                          <p>Original deadline: <strong>{dateLabel(task.original_due_date)}</strong></p>
                          <p>Suggested deadline: <strong>{task.suggested_due_date ? dateLabel(task.suggested_due_date) : 'No valid date available'}</strong></p>
                          <small>{task.explanation}</small>
                        </div>
                      ))}</div>
                      <p className="recovery-limitation">{recoveryResult.notice}</p>
                      {recoveryResult.can_apply && <button className="recovery-apply-button" type="button" onClick={applyTimelineRecovery}>Apply suggested dates</button>}
                    </div>}
                  </section>
                </article>
                <article className="vendors-card">
                  <div className="card-title-row"><div><span className="section-kicker">YOUR SHORTLIST</span><h3>Who to get in touch with.</h3></div><MapPin size={19} className="muted-icon" /></div>
                  <div className="vendor-grid">{plan.vendor_categories.map((vendor) => <div className="vendor-item" key={vendor.category}><span className="vendor-check"><Check size={13} /></span><span>{vendor.category}</span>{vendor.essential && <small>ESSENTIAL</small>}</div>)}</div>
                  <p className="vendor-note">{plan.vendor_notice}</p>
                </article>
                <article className="shock-card">
                  <div className="card-title-row">
                    <div><span className="section-kicker">PLAN FOR THE UNEXPECTED</span><h3>Budget Shock Simulator</h3></div>
                    <span className="guard-icon"><Sparkles size={17} /></span>
                  </div>
                  <p className="shock-intro">See how one surprise could change your plan. This simulation is separate from your original plan.</p>
                  <form className="shock-form" onSubmit={simulateBudgetShock}>
                    <label>What changed?
                      <select value={shockType} onChange={(e) => { setShockType(e.target.value); setShockResult(null); setShockError(''); }}>
                        <option value="category_percentage">A category costs more</option>
                        <option value="additional_guests">More guests are coming</option>
                        <option value="fixed_expense">An unexpected expense</option>
                      </select>
                    </label>
                    {shockType === 'category_percentage' && <>
                      <label>Budget category
                        <select required value={shockCategory} onChange={(e) => setShockCategory(e.target.value)}>
                          {plan.budget.allocations.map((item) => <option key={item.category} value={item.category}>{item.category}</option>)}
                        </select>
                      </label>
                      <label>Increase (%)<input required type="number" min="0.01" max="500" step="0.01" value={shockPercent} onChange={(e) => setShockPercent(e.target.value)} /></label>
                    </>}
                    {shockType === 'additional_guests' && <label>Additional guests<input required type="number" min="1" max="10000" value={shockAdditionalGuests} onChange={(e) => setShockAdditionalGuests(e.target.value)} /></label>}
                    {shockType === 'fixed_expense' && <div className="shock-field-row">
                      <label>Expense name<input required maxLength="80" value={shockExpenseLabel} onChange={(e) => setShockExpenseLabel(e.target.value)} placeholder="e.g. Extra transport" /></label>
                      <label>Expense amount (₹)<input required type="number" min="1" max="10000000" value={shockExpenseAmount} onChange={(e) => setShockExpenseAmount(e.target.value)} /></label>
                    </div>}
                    <button className="generate-button shock-submit" type="submit" disabled={shockLoading}>
                      {shockLoading ? <><span className="spinner" /> Simulating…</> : <><Sparkles size={16} /> Simulate budget shock <ArrowRight size={15} /></>}
                    </button>
                  </form>
                  {shockError && <div className="error-message" role="alert">{shockError}</div>}
                  {shockResult && <div className="shock-results" aria-live="polite">
                    <div className="shock-summary">
                      <div><span>Original plan cost</span><strong>{money(shockResult.original_allocation_total_inr)}</strong></div>
                      <div><span>Simulated cost</span><strong>{money(shockResult.projected_total_inr)}</strong></div>
                      <div><span>Available budget</span><strong>{money(shockResult.available_budget_inr)}</strong></div>
                      <div><span>Projected cost / guest</span><strong>{money(shockResult.projected_cost_per_guest_inr)}</strong></div>
                    </div>
                    <p className="shock-description">{shockResult.shock_description}</p>
                    {shockResult.projected_gap_inr > 0
                      ? <div className="shock-gap" role="status"><strong>Projected budget gap: {money(shockResult.projected_gap_inr)}</strong></div>
                      : <div className="budget-ok shock-gap-ok"><CheckCircle2 size={15} /> Simulated costs remain within the original budget.</div>}
                    <div className="shock-comparison">
                      <h4>Category comparison</h4>
                      <div className="shock-table-wrap"><table>
                        <thead><tr><th scope="col">Category</th><th scope="col">Original</th><th scope="col">Simulated</th><th scope="col">Change</th></tr></thead>
                        <tbody>{shockResult.category_comparison.map((item) => <tr key={item.category}><th scope="row">{item.category}</th><td>{money(item.baseline_amount_inr)}</td><td>{money(item.projected_amount_inr)}</td><td>{item.change_inr > 0 ? '+' : ''}{money(item.change_inr)}</td></tr>)}</tbody>
                      </table></div>
                    </div>
                    <div className={`recovery-result ${shockResult.recovery.fits_within_budget ? 'recovery-success' : 'recovery-unresolved'}`} role={shockResult.recovery.fits_within_budget ? 'status' : 'alert'}>
                      <strong>{shockResult.recovery.fits_within_budget
                        ? (shockResult.projected_gap_inr > 0 ? 'Recovery proposal fits the budget' : 'No recovery needed')
                        : `Recovery is insufficient: the plan still exceeds the budget by ${money(shockResult.recovery.remaining_shortfall_inr)}`}</strong>
                      {shockResult.recovery.fits_within_budget && <div className="recovery-success-totals">
                        <span>Original projected cost <strong>{money(shockResult.projected_total_inr)}</strong></span>
                        <span>Total proposed savings <strong>{money(shockResult.recovery.total_savings_inr)}</strong></span>
                        <span>Final proposed total <strong>{money(shockResult.recovery.proposed_total_inr)}</strong></span>
                        <span>Budget remaining <strong>{money(shockResult.recovery.remaining_budget_inr)}</strong></span>
                      </div>}
                      <p>{shockResult.recovery.notice}</p>
                      {shockResult.recovery.savings_breakdown.length > 0
                        ? <ul>{shockResult.recovery.savings_breakdown.map((saving) => <li key={saving.category}>{saving.category}: {money(saving.before_reduction_amount_inr)} → {money(saving.revised_amount_inr)} (save {money(saving.reduction_inr)}; up to {saving.maximum_reduction_percent}% of original allocation)</li>)}</ul>
                        : <p>No eligible discretionary reductions are available.</p>}
                      <div className="recovery-total">Proposed total: <strong>{money(shockResult.recovery.proposed_total_inr)}</strong>
                        {shockResult.recovery.total_savings_inr > 0 && <> · Savings: <strong>{money(shockResult.recovery.total_savings_inr)}</strong></>}
                      </div>
                    </div>
                    <p className="estimate-note">{shockResult.estimate_notice}{shockType === 'additional_guests' && ' Additional guests use the baseline plan’s rounded average cost per guest.'}</p>
                  </div>}
                </article>
                <article className="whatif-card">
                  <div><span className="section-kicker">A QUICK WHAT-IF</span><h3>Plans change. That’s okay.</h3><p>Adjust your guest count or budget and see a fresh estimate.</p></div>
                  <form onSubmit={regenerateWhatIf} className="whatif-form">
                    <label>Guests<input type="number" min="1" max="10000" placeholder={plan.event.guest_count} value={adjustGuests} onChange={(e) => setAdjustGuests(e.target.value)} /></label>
                    <label>Budget (₹)<input type="number" min="1000" max="10000000" step="1000" placeholder={plan.budget.budget_inr} value={adjustBudget} onChange={(e) => setAdjustBudget(e.target.value)} /></label>
                    <button disabled={loading} aria-label="Regenerate plan"><ArrowRight size={18} /></button>
                  </form>
                </article>
              </>}
            </section>
          </div>
        </section>
      </main>
      <footer><a className="brand footer-brand" href="#"><span className="brand-mark"><PartyPopper size={16} /></span><span>eventease<span className="brand-ai">.ai</span></span></a><span>Thoughtful plans for the moments that matter.</span><span>PromptWars Virtual 2026 · Challenge 05</span></footer>
    </div>
  );
}

export default App;
