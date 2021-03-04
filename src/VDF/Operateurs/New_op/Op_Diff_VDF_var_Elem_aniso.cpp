/****************************************************************************
* Copyright (c) 2021, CEA
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*****************************************************************************/
//////////////////////////////////////////////////////////////////////////////
//
// File:        Op_Diff_VDF_var_Elem_aniso.cpp
// Directory:   $TRUST_ROOT/src/VDF/Operateurs/New_op
// Version:     /main/13
//
//////////////////////////////////////////////////////////////////////////////

#include <Op_Diff_VDF_var_Elem_aniso.h>

Implemente_instanciable_sans_constructeur(Op_Diff_VDF_var_Elem_aniso,"Op_Diff_VDFANISOTROPE_var_P0_VDF",Op_Diff_VDF_Elem_base2);
implemente_It_VDF_Elem(Eval_Diff_VDF_var_Elem_aniso)

Sortie& Op_Diff_VDF_var_Elem_aniso::printOn(Sortie& s ) const
{
  return s << que_suis_je() ;
}

Entree& Op_Diff_VDF_var_Elem_aniso::readOn(Entree& s )
{
  return s ;
}

//
// Fonctions inline de la classe Op_Diff_VDF_var_Elem_aniso
//
// Description:
// constructeur
Op_Diff_VDF_var_Elem_aniso::Op_Diff_VDF_var_Elem_aniso() :
  Op_Diff_VDF_Elem_base2(It_VDF_Elem(Eval_Diff_VDF_var_Elem_aniso)())
{
}
