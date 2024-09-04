/****************************************************************************
* Copyright (c) 2024, CEA
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

#ifndef Elem_geom_included
#define Elem_geom_included

#include <Elem_geom_base.h>
#include <TRUST_Deriv.h>

/*! @brief Classe Elem_geom Classe generique de la hierarchie des elements geometriques, un objet
 *
 *      Elem_geom peut referencer n'importe quel objet derivant de
 *      Elem_geom_base.
 *      La plupart des methodes appellent les methodes de l'objet Elem_geom_base
 *      sous-jacent via la methode valeur() declaree grace a la macro
 *
 */
template <typename _SIZE_>
class Elem_geom_32_64 : public OWN_PTR(Elem_geom_base_32_64<_SIZE_>)
{
  Declare_instanciable_32_64(Elem_geom_32_64);
};

using Elem_geom = Elem_geom_32_64<int>;
using Elem_geom_64 = Elem_geom_32_64<trustIdType>;

#endif
