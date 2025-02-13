import nbformat as nbf
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser(description="Create jupyter notebook")
parser.add_argument('-n', '--namefile', required=True,
                    metavar="File", help="jupyter notebook name")
parser.add_argument('-d', '--datafile', nargs="+", required=True,
                    metavar="File", help="List of data files")
parser.add_argument('-t', '--tutorial', required=False, type=int,
                    help="print more examples")
args = parser.parse_args()

Files = args.datafile

nb = nbf.v4.new_notebook()

title = f"# {args.namefile} test case\n"

intro = "## Introduction\n \n Validation made by: " + os.environ.get("USER") + "\n\n"
intro += " Report generated " + datetime.now().strftime("%d/%m/%Y")

description = "### Description\n \n"
description += "Sketch of a classic jupyter validation form with main example of validation.\n\n"
description += "For additional information, please consult validation form documentation with <code>Run_fiche -doc</code> and report to the more detailed tutorial in $TRUST_ROOT/Validation/Rapports_automatiques/Verification/SampleFormJupyter."

binary = os.environ.get("exec", "[UNKNOWN]")
version = os.environ.get("TRUST_VERSION", "[UNKNOWN]")

origin = os.getcwd()
path = os.path.join(origin, "build")

parameters = "from trustutils import run\n\n"
parameters += f"run.TRUST_parameters(\"{version}\")"

testCases = "from trustutils import run\n\n"
testCases += "run.reset()\n"

for datafile in Files:
    dirname_, basename_ = os.path.split(datafile)
    if dirname_ == "":
        dirname_ = "."
    testCases += f"run.addCase(\"{dirname_}\",\"{basename_}\",nbProcs=1)\n"
testCases += "run.printCases()\n"
testCases += "run.runCases()"

if args.tutorial == 1:
    exampletest = Files[0].split('.data')[0]

    graph_t = "## Visualization of the probes evolution (point and segment example)"
    graph = "from trustutils import plot \n \n"
    graph += "graph1=plot.Graph(\"Sonde point\",size=[15,8])\n"
    graph += f"graph1.addPoint(\"{exampletest}_SONDE_PRESSION.son\",compo=0,marker=\"+\")\n\n"
    graph += "graph2=plot.Graph(\"Sonde segment\",size=[15,8])\n"
    graph += f"graph2.addSegment(\"{exampletest}_SONDE_VITESSE.son\",compo=0,marker=\"-x\")\n"

    visit_t = "## Visualization via VisIt (Pressure field example)"
    visit = "from trustutils import visit\n\n"
    visit += f"fig=visit.Show(\"{exampletest}.lata\",\"Pseudocolor\",\"PRESSION_ELEM_dom\",nX=1,nY=1,mesh=\"dom\",title=\"\",time=-1)\n"
    visit += "fig.plot()"

    tab_t = "## Table Creation"
    tab = "from trustutils import plot\n\n"
    tab += "tab = plot.Table(['column1','column2'])\n"
    tab += "data = plot.loadText(\"file.dat\",transpose=False)\n"
    tab += "tab.addLigne([[data[0],data[1]]],\"label\")\n"
    tab += "display(tab)"

    data_t = "## Data"
    data = f"run.dumpDataset(\"{exampletest}.data\")"

perf_t = "## Computer Performance"
perf = "run.tablePerf()"

nb['cells'] = [nbf.v4.new_markdown_cell(title),
               nbf.v4.new_markdown_cell(intro),
               nbf.v4.new_markdown_cell(description),
               nbf.v4.new_code_cell(parameters),
               nbf.v4.new_code_cell(testCases)]

if args.tutorial == 1:
    nb['cells'] += [nbf.v4.new_markdown_cell(graph_t),
                    nbf.v4.new_code_cell(graph),
                    nbf.v4.new_markdown_cell(visit_t),
                    nbf.v4.new_code_cell(visit),
                    nbf.v4.new_markdown_cell(tab_t),
                    nbf.v4.new_code_cell(tab)]

nb['cells'] += [nbf.v4.new_markdown_cell(perf_t),
                nbf.v4.new_code_cell(perf)]

if args.tutorial == 1:
    nb['cells'] += [nbf.v4.new_markdown_cell(data_t),
                    nbf.v4.new_code_cell(data)]

nbf.write(nb, f"{args.namefile}.ipynb")
