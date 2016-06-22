/****************************************************************************
* Copyright (c) 2015 - 2016, CEA
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
// File:        DoubleTrav.cpp
// Directory:   $TRUST_ROOT/src/Kernel/Math
// Version:     /main/10
//
//////////////////////////////////////////////////////////////////////////////
//
// WARNING: DO NOT EDIT THIS FILE! Only edit the template file DoubleTrav.cpp.P
//
//
#include <DoubleTrav.h>
#include <math.h>

Implemente_instanciable_sans_constructeur_ni_destructeur(DoubleTrav,"DoubleTrav",DoubleTab);


#define zero_double 0.
#define zero_int 0
#define zero_long 0
// Description: Appel a DoubleTab::printOn
Sortie& DoubleTrav::printOn(Sortie& os) const
{
  DoubleTab::printOn(os);
  return os;
}

// Description: Appel a DoubleTab::readOn
Entree& DoubleTrav::readOn(Entree& is)
{
  DoubleTab::readOn(is);
  return is;
}


// Description:
DoubleTrav::DoubleTrav()
{
  set_mem_storage(TEMP_STORAGE);
}

// Description:
DoubleTrav::DoubleTrav(int n)
{
  set_mem_storage(TEMP_STORAGE);
  resize(n);
}

// Description:
DoubleTrav::DoubleTrav(int n1, int n2)


{
  set_mem_storage(TEMP_STORAGE);
  resize(n1, n2);
}


// Description:
DoubleTrav::DoubleTrav(int n1, int n2, int n3)
{
  set_mem_storage(TEMP_STORAGE);
  resize(n1, n2, n3);
}

// Description:
DoubleTrav::DoubleTrav(int n1, int n2, int n3, int n4)
{
  set_mem_storage(TEMP_STORAGE);
  resize(n1, n2, n3, n4);
}

// Description:
//  ATTENTION: construit un tableau de meme taill et de meme structure
//   (espaces virtuels), mais initialise avec ZERO !!!
DoubleTrav::DoubleTrav(const DoubleTab& tab)
{
  set_mem_storage(TEMP_STORAGE);
  DoubleTab::copy(tab, Array_base::NOCOPY_NOINIT);
  DoubleTab::operator=(zero_double);
}

// Description:
// Constructeur par copie
//  ATTENTION: construit un tableau de meme taill et de meme structure
//   (espaces virtuels), mais initialise avec ZERO !!!
DoubleTrav::DoubleTrav(const DoubleTrav& tab):DoubleTab()
{
  set_mem_storage(TEMP_STORAGE);
  DoubleTab::copy(tab, Array_base::NOCOPY_NOINIT);
  DoubleTab::operator=(zero_double);
}

// Description:
// Constructeur par copie
//  ATTENTION: construit un tableau de meme taill et de meme structure
//   (espaces virtuels), mais initialise avec ZERO !!!
DoubleTrav::DoubleTrav(const DoubleVect& tab)
{
  set_mem_storage(TEMP_STORAGE);
  set_line_size_(tab.line_size());
  DoubleTab::resize(tab.size_array(), Array_base::NOCOPY_NOINIT);
  set_md_vector(tab.get_md_vector());
  DoubleTab::operator=(zero_double);
}

// Description:
// Operateur copie (on ne veut pas l'operateur par defaut)
DoubleTrav& DoubleTrav::operator=(const DoubleTrav& tab)
{
  DoubleTab::operator=(tab);
  return *this;
}

// Description:
//  Operateur copie d'un tableau (copie structure ET contenu)
DoubleTrav& DoubleTrav::operator=(const DoubleTab& tab)
{
  // ATTENTION: note aux programmeurs
  //  La declaration de cet operateur est indispensable, sinon
  //   DoubleTab b;
  //   DoubleTrav a;
  //   a = b
  //  est traduit en
  //   DoubleTrav tmp(b); // copie la structure mais met les valeurs a zero
  //   a.operator=(tmp);
  DoubleTab::operator=(tab);
  return *this;
}
// Description:
//  Operateur copie d'un tableau (copie structure ET contenu)
DoubleTrav& DoubleTrav::operator=(const DoubleVect& tab)
{
  // ATTENTION: note aux programmeurs
  //  La declaration de cet operateur est indispensable, sinon
  //   DoubleTab b;
  //   DoubleTrav a;
  //   a = b
  //  est traduit en
  //   DoubleTrav tmp(b); // copie la structure mais met les valeurs a zero
  //   a.operator=(tmp);
  DoubleTab::operator=(tab);
  return *this;
}
// Description:
// Comme on surcharge l'operateur copie, il faut redefinir celui-la aussi.
DoubleTrav& DoubleTrav::operator=(double d)
{
  DoubleTab::operator=(d);
  return *this;
}

// Description:
// Precondition:
// Parametre:
//    Signification:
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour:
//    Signification:
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
int DoubleTrav_test()
{
  int j,i=1;
  while(i)
    {
      Cout << "imput of i j (0 0 for => end)";
      cin >> i >> j;
      Cout << finl;
      {
        DoubleTrav x(i, j);
        Cout << x.size()<< finl;
      }
    }
  {
    DoubleTab y(10, 10, (double)1);
    DoubleTrav x(y);
    x(5,5) = (double)1;
    DoubleTab z(x);
    Cout << z(5,5)<< " " << z(3,3) << finl;
  }
  return 1;
}
#undef zero_double
#undef zero_int
#undef zero_long

