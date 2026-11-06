"""Local option-pricing application. Start with: python app.py."""
import csv
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import reproduce as model


def calculate(option, paths, strike, intervals, grid, seed):
    """Shared application engine; returns prices and clearly separated errors."""
    result = []
    if option == 'Arithmetic Asian call':
        for method in ['Standard', 'Geometric CV']:
            start=time.perf_counter()
            if method=='Standard':
                y,_=model.asian_sample(paths,seed,False)
            else:
                beta=model.asian_beta(seed+10000000)
                a,g=model.asian_sample(paths,seed)
                y=a-beta*(g-model.geometric_price())
            elapsed=time.perf_counter()-start
            st=model.stats(y)
            result.append(dict(method=method,price=st['price'],se=st['se'],
                ci_low=st['ci_low'],ci_high=st['ci_high'],absolute_error='',seconds=elapsed))
        return None,result
    reference = model.digital_exact(strike) if option == 'Digital call' else model.barrier_ref()[0]
    if option == 'Digital call':
        methods = ['Standard', 'Antithetic', 'IS']
        samples = lambda method: model.digital_sample(paths, method, seed, strike)
    else:
        methods = ['Discrete monitoring', 'Brownian bridge']
        samples = lambda method: model.barrier_sample(paths, intervals, seed, method == 'Brownian bridge')
    for method in methods:
        start = time.perf_counter()
        st = model.stats(samples(method))
        result.append(dict(method=method, price=st['price'], se=st['se'],
                           ci_low=st['ci_low'], ci_high=st['ci_high'],
                           absolute_error=abs(st['price']-reference), seconds=time.perf_counter()-start))
    kind = 'digital' if option == 'Digital call' else 'barrier'
    for scheme in ['Explicit', 'Implicit', 'CN', 'Rannacher']:
        steps = int(np.ceil((model.SIG**2*(grid-1)**2+model.R)/.9)) if scheme == 'Explicit' else grid
        start = time.perf_counter()
        ans = model.fd(kind, grid, steps, scheme, strike=strike if kind == 'digital' else 100.)
        result.append(dict(method=f'{scheme} FD (N={steps})', price=ans['price'], se='',
                           ci_low='', ci_high='', absolute_error=abs(ans['price']-reference),
                           seconds=time.perf_counter()-start))
    return reference, result


class App:
    def __init__(self, root):
        self.root = root
        root.title('FMAT3888 | Option pricing | SID 20260914')
        root.geometry('1120x680')
        root.minsize(950, 580)
        style = ttk.Style(root)
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        style.configure('Title.TLabel', font=('Segoe UI', 21, 'bold'))
        style.configure('Treeview', rowheight=30)
        frame = ttk.Frame(root, padding=22)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='Option pricing laboratory', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, text='FMAT3888  |  SID 20260914  |  S0 = 100, r = 0.05, sigma = 0.25, T = 1').pack(anchor='w', pady=(4,18))
        controls = ttk.LabelFrame(frame, text='Experiment', padding=12)
        controls.pack(fill='x')
        self.option=tk.StringVar(value='Digital call')
        self.paths=tk.StringVar(value='100000')
        self.strike=tk.StringVar(value='140')
        self.intervals=tk.StringVar(value='16')
        self.grid=tk.StringVar(value='260')
        self.seed=tk.StringVar(value='20260914')
        entries = [('Contract', self.option), ('MC payoff budget',self.paths), ('Digital strike',self.strike),
                   ('Barrier intervals',self.intervals), ('FD space intervals',self.grid), ('Random seed',self.seed)]
        for col,(label,var) in enumerate(entries):
            ttk.Label(controls,text=label).grid(row=0,column=col,sticky='w',padx=5)
            if col==0:
                widget=ttk.Combobox(controls,textvariable=var,values=['Digital call','Up-and-out call','Arithmetic Asian call'],state='readonly',width=21)
                widget.bind('<<ComboboxSelected>>', self.selection_changed)
            else:
                widget=ttk.Entry(controls,textvariable=var,width=15)
                if col==2:self.strike_entry=widget
                if col==3:self.intervals_entry=widget
                if col==4:self.grid_entry=widget
            widget.grid(row=1,column=col,sticky='ew',padx=5,pady=5)
            controls.columnconfigure(col,weight=1)
        actions=ttk.Frame(frame); actions.pack(fill='x',pady=14)
        self.run=ttk.Button(actions,text='Calculate and compare',command=self.start)
        self.run.pack(side='left')
        self.export=ttk.Button(actions,text='Export results as CSV',command=self.save,state='disabled')
        self.export.pack(side='left',padx=8)
        self.status=tk.StringVar(value='Ready. Select a contract and run an experiment.')
        ttk.Label(frame,textvariable=self.status,wraplength=1040).pack(anchor='w',pady=(0,10))
        columns=('method','price','se','ci','error','time')
        self.table=ttk.Treeview(frame,columns=columns,show='headings',height=8)
        for col,title,width in zip(columns,['Method','Price','MC standard error','MC 95% interval','Absolute error','Seconds'],[230,105,130,210,125,85]):
            self.table.heading(col,text=title); self.table.column(col,width=width,anchor='w' if col=='method' else 'e')
        self.table.pack(fill='both',expand=True)
        note=('Digital: exact terminal simulation; FD uses cell-averaged payoff on [0,400]. Antithetic uses half as many independent pairs.\n'
              'Barrier: K = 100 and H = 130, continuously monitored. Endpoint-only MC has monitoring bias; its interval covers sampling error only.\n'
              'FD: implicit/CN/Rannacher use N = M; explicit N is chosen for stability. Rannacher starts with two implicit half-steps.\n'
              'Asian (Extension B): K = 100; 12 monthly dates excluding S0. Geometric CV timings include an independent 20,000-path pilot.\n'
              'The arithmetic Asian has no exact reference here: its absolute error is unavailable. Compare standard errors and confidence intervals.')
        ttk.Label(frame,text=note,wraplength=1040,justify='left').pack(anchor='w',pady=(16,0))
        self.messages=queue.Queue();self.results=[]
        self.selection_changed()
        root.after(100,self.poll)

    def selection_changed(self,event=None):
        digital=self.option.get()=='Digital call'
        barrier=self.option.get()=='Up-and-out call'
        self.strike_entry.configure(state='normal' if digital else 'disabled')
        self.intervals_entry.configure(state='normal' if barrier else 'disabled')
        self.grid_entry.configure(state='disabled' if self.option.get()=='Arithmetic Asian call' else 'normal')

    def start(self):
        try:
            n=int(self.paths.get());k=float(self.strike.get()) if self.option.get()=='Digital call' else 100.
            m=int(self.intervals.get()) if self.option.get()=='Up-and-out call' else (12 if self.option.get()=='Arithmetic Asian call' else 1)
            grid=int(self.grid.get()) if self.option.get()!='Arithmetic Asian call' else 260
            seed=int(self.seed.get())
            if not(1000<=n<=2000000 and n%2==0):raise ValueError('Use an even MC budget between 1,000 and 2,000,000.')
            if not(80<=k<=220):raise ValueError('Use a digital strike between 80 and 220.')
            if not(1<=m<=256 and n*m<=30000000):raise ValueError('Use 1-256 barrier intervals and at most 30 million path increments.')
            if not(40<=grid<=520):raise ValueError('Use 40-520 spatial intervals for this interactive calculation.')
            if not(0<=seed<2**32):raise ValueError('Use a nonnegative seed below 2^32.')
        except ValueError as exc:
            messagebox.showerror('Check inputs',str(exc));return
        self.run.configure(state='disabled');self.export.configure(state='disabled')
        asian=self.option.get()=='Arithmetic Asian call'
        self.status.set('Calculating standard MC and geometric control variate...' if asian else 'Calculating MC and four FD schemes...')
        self.configuration=dict(contract=self.option.get(),paths=n,strike=k,barrier=130 if self.option.get()=='Up-and-out call' else '',intervals=m,grid='' if asian else grid,seed=seed)
        def worker():
            try:self.messages.put(('ok',calculate(self.configuration['contract'],n,k,m,grid,seed)))
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            status,payload=self.messages.get_nowait()
            self.run.configure(state='normal')
            if status=='error':
                self.status.set('Calculation failed.');messagebox.showerror('Calculation error',payload)
            else:
                ref,self.results=payload
                self.table.delete(*self.table.get_children())
                for row in self.results:
                    ci=f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]" if row['se']!='' else '-'
                    error=f"{row['absolute_error']:.3g}" if row['absolute_error']!='' else '-'
                    self.table.insert('', 'end',values=(row['method'],f"{row['price']:.8f}",f"{row['se']:.6g}" if row['se']!='' else '-',ci,error,f"{row['seconds']:.4f}"))
                if ref is None:
                    vr=(self.results[0]['se']/self.results[1]['se'])**2
                    self.status.set(f'Asian complete. Estimated variance reduction: {vr:.1f}x. Geometric control mean: {model.geometric_price():.8f}. Arithmetic exact error unavailable.')
                else:
                    self.status.set(f"Complete. Reference price: {ref:.10f}. Timings are single runs; the report uses medians of three runs.")
                self.export.configure(state='normal')
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def save(self):
        target=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='option_results.csv',filetypes=[('CSV','*.csv')])
        if target:
            try:
                rows=[dict(**self.configuration,**row) for row in self.results]
                with Path(target).open('w',newline='',encoding='utf-8-sig') as file:
                    writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
                self.status.set(f'Results saved to {target}')
            except OSError as exc:messagebox.showerror('Export failed',str(exc))


if __name__=='__main__':
    window=tk.Tk();App(window);window.mainloop()
