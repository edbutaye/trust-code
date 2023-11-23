#!/usr/bin/python

"""
This script generates the RST reference documentation for all the TRUST keywords.
It uses only the generated Python module (this is the easiest since the module contains all the 
relevant docstrings and inheritance relationships)
"""

import os, sys

def no_tru(s):
  """ Remove the "_Tru" suffix of a class name ... """
  return s.replace("_Tru", "")

class TRUSTDocGenerator:
  # For those classes, we want a specific type display, and we do not want a full entry in the
  # reference manual (those are basic root types):
  TYPE_MAP =  {"chaine": "string",
               "floattant": "double",
               "rien": "flag",
               "entier": "int",
               "list": "list",
               "listchaine": "string list",
               "listchainef": "string list",
               "listentierf": "int list - size is dimension",
               "listfloattantf": "double list - size is dimension",
               "listentier": "int list",
               "listfloattant": "double list",
               "listf": "double list",
               }

  def __init__(self, py_mod=None):
    """ @param py_mod (str): path to the automatically generated Python module
    """
    self.py_mod = py_mod
    if py_mod is None:
      if "TRUST_ROOT" not in os.environ:
        print("Generated module not specified and TRUST_ROOT not defined! Exiting.")
        sys.exit(-1)
      self.py_mod = os.path.join(os.environ["TRUST_ROOT"], "Outils", "trustpy", "install", "generated", "trustpy_gen.py")

  def get_top_parent(self, c):
    """ Get ultimate parent class """
    from trustpy.baseTru import ConstrainBase_Tru, ListOfBase_Tru
    # Recurse of course
    def get_parent(cls):
      bases = cls.__bases__
      if len(bases) != 1:
        return cls
      if bases[0] is ConstrainBase_Tru or bases[0] is ListOfBase_Tru:
        return cls
      return get_parent(bases[0])

    return get_parent(c)

  def extractTypeAndDesc(self, s):
    """ From a string like
       '[chaine(into=["lml","lata","lata_v2","med"])] generated file post_med.data use format'
    produces two strings, one for the type, and one for the desc
       'string into ["lml","lata","lata_v2","med"]'
       'generated file post_med.data use format'
     """
    # A bit ugly, but works:
    i = s.index("]")
    if s.startswith("[chaine(into=") or s.startswith("[entier(into="):
      i += s[i+1:].index("]")
    typ, desc = s[1:i], s[i+2:]
    # Beautify type...
    if typ.startswith("ref_"):
      typ = typ[4:]
    if not (typ in self.TYPE_MAP or "into=" in typ or "max=" in typ):
      typ_with_ref = f":ref:`{typ}`"
    else:
      if "into=" in typ:
        typ = typ.replace('=', ' ').replace("chaine", "string").replace("entier", "int").replace("(", " ")
      typ_with_ref = self.TYPE_MAP.get(typ, typ)
    # Description must not contain any line return - not supported in item of a list
    desc = desc.replace("\n", " ")
    return typ_with_ref, desc

  def doc_single(self, c, parent):
    """ Generate full RST string for a single keyword """
    # Main name and synonyms
    nam = no_tru(c.__name__)
    s = f".. _{nam}:\n\n"
    s += f"**{nam}**\n"
    s += "-" * (len(nam)+4)
    if len(c._synonyms):
      syno = [s.replace("_Tru", "") for s in c._synonyms]
      s += "\n\n"
      s += "**Synonyms:** %s\n" % ", ".join(syno)
    par = ":ref:`%s`" % no_tru(parent.__name__)
    # Inheritance
    s += "\n**Inherits from:** %s \n\n" % par
    # Core description
    core_doc = c.__doc__
    t = core_doc.split("\n")
    t2 = [tt.lstrip() for tt in t]
    s += "\n".join(t2)
    # If this is a list we can stop here, no attributes
    if not hasattr(c, "_attributesList"):
      return s
    if len(c._attributesList) != 0:
      s += "\nParameters are:\n\n"
    # Attributes, in the same order as found in the Python generated module:
    for a, typ in c._attributesList:
      asyno = c._attributesSynonyms.get(a, [])
      all_att = " | ".join([a]+asyno)
      hd_a = ""
      if a in c._optionals:
        hd_a += f"- **[{all_att}]** "
      else:
        hd_a += f"- **{all_att}** "
      hlp = c._helpDict[a][0]
      typ, dsc = self.extractTypeAndDesc(hlp)
      s += f"{hd_a} (*type:* {typ}) {dsc}\n\n"
    return s

  def generate_doc(self, out_dir):
    """ Generate MD documentation for all classes in TRUST """
    # Import generated module:
    from trustpy.trust_utilities import import_generated_module
    ze_mod = import_generated_module(self.py_mod)

    # Identify top classes in the module - discard automatic ones
    all_cls = ze_mod.packagespy_classes
    top_cls = {}
    # Keep valid classes and find their parent class:
    for c in all_cls:
      p = self.get_top_parent(c)
      pn, cn = no_tru(p.__name__), no_tru(c.__name__)
      # Classes in self.TYPE_MAP (and their children) should simply be skipped:
      if pn in self.TYPE_MAP or cn in self.TYPE_MAP:
        # print("skiping ", cn)
        continue
      # Classes starting with upper case should also be skipped (manual classes from baseTru.py)
      if cn[0].isupper() or pn[0].isupper():
        # print("skiping ", cn)
        continue
      # Otherwise class is registered:
      top_cls[c] = p

    # Unique parents, sorted alphabetically
    prt = list(set(top_cls.values()))
    prt.sort(key=lambda c: c.__name__)
    # Invert dict top_cls:
    inv_top = {}
    for k, v in top_cls.items():
      inv_top.setdefault(v, []).append(k)
    # Single string:
    single_s = ""
    # One MD file per parent:
    for j, p in enumerate(prt):
      lst = inv_top[p]
      lst.sort(key=lambda c: c.__name__)
      par = no_tru(p.__name__)
      s = f"**Keywords derived from {par}**\n"
      s += "=" * (len(s)-1) + "\n\n"  # +4 because of '**'
      for i, c in enumerate(lst):
        if i != 0:
          # Horizontal line
          s += "\n----\n\n"
        else:
          # At first iteration, generate parent class doc, if not there already
          if p not in lst:
            s += self.doc_single(p, p)
            s += "\n----\n\n"
        s += self.doc_single(c, p)
      fname = os.path.join(out_dir, no_tru(p.__name__) + ".rst")
      with open(fname, "w") as f:
        f.write(s)
      if j != 0:
        # Horizontal line
        single_s += "\n----\n\n"
      single_s += s

    # One big file with everything:
    fname_single = os.path.join(out_dir, "all.rst")
    with open(fname_single, "w") as f:
      f.write(single_s)

if __name__ == "__main__":
  pm = None
  if len(sys.argv) < 3:
    print("Usage: ./gen_doc.py <generated_module> <out_directory>")
    sys.exit(-1)
  pm, out_dir = sys.argv[1], sys.argv[2]
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  tdg = TRUSTDocGenerator(py_mod=pm)
  #s = """[chaine(into=["lml","lata","lata_v2","med"])] generated file post_med.data use format"""
  #print(tdg.extractTypeAndRef(s))
  tdg.generate_doc(out_dir)
