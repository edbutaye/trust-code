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
// File:        Sortie.cpp
// Directory:   $TRUST_ROOT/src/Kernel/Utilitaires
// Version:     /main/portage_windows/1
//
//////////////////////////////////////////////////////////////////////////////
//
// WARNING: DO NOT EDIT THIS FILE! Only edit the template file Sortie.cpp.P
//

#include <Sortie.h>
#include <Objet_U.h>
#include <Separateur.h>
#include <assert.h>
#include <iostream>

const Separateur finl(Separateur::ENDL);
const Separateur space(Separateur::SPACE);

// Constructeurs
Sortie::Sortie()
{
  bin_=0;
  ostream_=0;
}

void Sortie::setf(IOS_FORMAT code)
{
  if(ostream_)
    ostream_->setf(code);
}

void Sortie::precision(int pre)
{
  if(ostream_)
    ostream_->precision(pre);
}

Sortie::Sortie(ostream& os)
{
  bin_=0;
  if(os.rdbuf())
    {
      ostream_ = new ostream(os.rdbuf());
    }
  else
    {
      Process::exit();
    }
}

Sortie::Sortie(const Sortie& os)
{
  bin_=0;
  if (os.has_ostream())
    {
      Cerr<<"we try to copy a Sortie with ostream !!!"<<finl;
      abort();
      ostream_ = new ostream(os.get_ostream().rdbuf());
    }
  else
    ostream_=0;
}

// Operateurs d'affectation
Sortie& Sortie::operator=(ostream& os)
{
  if(ostream_)
    delete ostream_;
  ostream_ = new ostream(os.rdbuf());
  return *this;
}

Sortie& Sortie::operator=(Sortie& os)
{
  if(ostream_)
    delete ostream_;
  if (os.has_ostream())
    {
      Cerr<<"we try to copy a Sortie with ostream !!!"<<finl;
      abort();
      ostream_ = new ostream(os.get_ostream().rdbuf());
    }
  else
    ostream_=0;
  return *this;
}

ostream& Sortie::get_ostream()
{
  return *ostream_;
}

const ostream& Sortie::get_ostream() const
{
  return *ostream_;
}

void Sortie::set_ostream(ostream* os)
{
  ostream_ = os;
}



// Description:
//  Methode de bas niveau pour ecrire un int ou flottant dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
Sortie& Sortie::operator<<(const int& ob)
{
  if(bin_)
    {
      /* Ecriture en mode binaire */
      ostream_->write((char *) & ob, sizeof(int));
    }
  else
    {
      /* Ecriture avec conversion en ascii */
      (*ostream_) << ob;
    }
  return *this;
}

// Description:
//  Methode de bas niveau pour ecrire un tableau d'ints ou reels dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
//  En ascii, on revient a la ligne chaque fois qu'on a ecrit "nb_col" valeurs et a la
//  fin du tableau.
//  Valeur de retour : ostream_->good()
int Sortie::put(const int * ob, int n, int nb_col)
{
  assert(n >= 0);
  if (bin_)
    {
      std::streamsize sz = sizeof(int);
      sz *= n;
      // Overflow checking :
      assert(sz / (std::streamsize)sizeof(int) == (std::streamsize)n);
      ostream_->write((const char *) ob, sz);
    }
  else
    {
      int j = nb_col;
      for (int i = 0; i < n; i++)
        {
          (*ostream_) << (ob[i]) << (' ');
          j--;
          if (j <= 0)
            {
              (*ostream_) << (endl);
              j = nb_col;
            }
        }
      // Si on n'a pas fini pas un retour a la ligne, en ajouter un
      if (j != nb_col && n > 0)
        (*ostream_) << (endl);
      ostream_->flush();
    }
  return ostream_->good();
}
#ifndef INT_is_64_
// Description:
//  Methode de bas niveau pour ecrire un int ou flottant dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
Sortie& Sortie::operator<<(const long& ob)
{
  if(bin_)
    {
      /* Ecriture en mode binaire */
      ostream_->write((char *) & ob, sizeof(long));
    }
  else
    {
      /* Ecriture avec conversion en ascii */
      (*ostream_) << ob;
    }
  return *this;
}

// Description:
//  Methode de bas niveau pour ecrire un tableau d'ints ou reels dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
//  En ascii, on revient a la ligne chaque fois qu'on a ecrit "nb_col" valeurs et a la
//  fin du tableau.
//  Valeur de retour : ostream_->good()
int Sortie::put(const long * ob, int n, int nb_col)
{
  assert(n >= 0);
  if (bin_)
    {
      std::streamsize sz = sizeof(long);
      sz *= n;
      // Overflow checking :
      assert(sz / (std::streamsize)sizeof(long) == (std::streamsize)n);
      ostream_->write((const char *) ob, sz);
    }
  else
    {
      int j = nb_col;
      for (int i = 0; i < n; i++)
        {
          (*ostream_) << (ob[i]) << (' ');
          j--;
          if (j <= 0)
            {
              (*ostream_) << (endl);
              j = nb_col;
            }
        }
      // Si on n'a pas fini pas un retour a la ligne, en ajouter un
      if (j != nb_col && n > 0)
        (*ostream_) << (endl);
      ostream_->flush();
    }
  return ostream_->good();
}
#endif
// Description:
//  Methode de bas niveau pour ecrire un int ou flottant dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
Sortie& Sortie::operator<<(const float& ob)
{
  if(bin_)
    {
      /* Ecriture en mode binaire */
      ostream_->write((char *) & ob, sizeof(float));
    }
  else
    {
      /* Ecriture avec conversion en ascii */
      (*ostream_) << ob;
    }
  return *this;
}

// Description:
//  Methode de bas niveau pour ecrire un tableau d'ints ou reels dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
//  En ascii, on revient a la ligne chaque fois qu'on a ecrit "nb_col" valeurs et a la
//  fin du tableau.
//  Valeur de retour : ostream_->good()
int Sortie::put(const float * ob, int n, int nb_col)
{
  assert(n >= 0);
  if (bin_)
    {
      std::streamsize sz = sizeof(float);
      sz *= n;
      // Overflow checking :
      assert(sz / (std::streamsize)sizeof(float) == (std::streamsize)n);
      ostream_->write((const char *) ob, sz);
    }
  else
    {
      int j = nb_col;
      for (int i = 0; i < n; i++)
        {
          (*ostream_) << (ob[i]) << (' ');
          j--;
          if (j <= 0)
            {
              (*ostream_) << (endl);
              j = nb_col;
            }
        }
      // Si on n'a pas fini pas un retour a la ligne, en ajouter un
      if (j != nb_col && n > 0)
        (*ostream_) << (endl);
      ostream_->flush();
    }
  return ostream_->good();
}
// Description:
//  Methode de bas niveau pour ecrire un int ou flottant dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
Sortie& Sortie::operator<<(const double& ob)
{
  if(bin_)
    {
      /* Ecriture en mode binaire */
      ostream_->write((char *) & ob, sizeof(double));
    }
  else
    {
      /* Ecriture avec conversion en ascii */
      (*ostream_) << ob;
    }
  return *this;
}

// Description:
//  Methode de bas niveau pour ecrire un tableau d'ints ou reels dans le stream.
//  Dans l'implementation de la classe de base, on ecrit dans ostream_.
//  En binaire on utilise ostream::write(), en ascii ostream::operato<<()
//  En ascii, on revient a la ligne chaque fois qu'on a ecrit "nb_col" valeurs et a la
//  fin du tableau.
//  Valeur de retour : ostream_->good()
int Sortie::put(const double * ob, int n, int nb_col)
{
  assert(n >= 0);
  if (bin_)
    {
      std::streamsize sz = sizeof(double);
      sz *= n;
      // Overflow checking :
      assert(sz / (std::streamsize)sizeof(double) == (std::streamsize)n);
      ostream_->write((const char *) ob, sz);
    }
  else
    {
      int j = nb_col;
      for (int i = 0; i < n; i++)
        {
          (*ostream_) << (ob[i]) << (' ');
          j--;
          if (j <= 0)
            {
              (*ostream_) << (endl);
              j = nb_col;
            }
        }
      // Si on n'a pas fini pas un retour a la ligne, en ajouter un
      if (j != nb_col && n > 0)
        (*ostream_) << (endl);
      ostream_->flush();
    }
  return ostream_->good();
}

Sortie& Sortie::operator <<(ostream& (*f)(ostream&))
{
  // Ca c'est pas genial, c'est pour permettre  "<< endl"
  // Probleme: ca ne marche pas si ostream_ == 0.
  if(ostream_ && !bin_)
    (*f)(*ostream_);
  return *this;
}

Sortie& Sortie::operator <<(Sortie& (*f)(Sortie&))
{
  (*f)(*this);
  return *this;
}

Sortie& Sortie::operator <<(ios& (*f)(ios&))
{
  if (ostream_)
    (*f)(*ostream_);
  return *this;
}

Sortie& Sortie::flush()
{
  //   get_ostream().flush();
  ostream_->flush();
  return *this;
}

Sortie& Sortie::operator<<(const Separateur& ob)
{
  if (bin_)
    {
      // En binaire on n'ecrit pas les separateurs
    }
  else
    {
      switch (ob.get_type())
        {
        case Separateur::ENDL:
          // On ecrit "\n" et pas endl...
          // C'est peut-etre une mauvaise idee
          //(*ostream_) << '\n';
          // GF pb sous windows avec ancienne ligne
          (*ostream_)<<endl;
          // Flush (important pour les fichiers de log)
          ostream_->flush();
          break;
        case Separateur::SPACE:
          (*ostream_) << ' ';
          break;
        }
    }
  return *this;
}

Sortie& Sortie::lockfile()
{
  Cerr<<"Sortie::lockfile() : we should not go through it !!"<<finl;
  Process::exit();
  return *this;
}

Sortie& Sortie::unlockfile()
{
  Cerr<<"Sortie::unlockfile() : we should not go through it !!"<<finl;
  Process::exit();
  return *this;
}

Sortie& Sortie::syncfile()
{
  Cerr<<"Sortie::syncfile() : we should not go through it !!"<<finl;
  Process::exit();
  return *this;
}

// Description:
//  Ecriture d'un objet ou d'une variable.
//  Dans cette implementation (et dans la plupart des classes derivees)
//  on appelle simplement ob.printOn (a l'exception de Sortie_Nulle)
//  Attention, si on veut que le flux puisse etre indifferemment ASCII ou BINAIRE,
//  il faut inserer "<< space <<"  ou "<< finl <<" pour separer les objets.
Sortie& Sortie::operator<<(const Objet_U& ob)
{
  ob.printOn(*this);
  return *this;
}

// Description:
// Ecriture d'une chaine de caracteres. Attention, pour pouvoir
// relire correctement la chaine en mode ascii, celle-ci ne doit
// pas contenir de separateur (ni espace, ni retour a la ligne, ...)
Sortie& Sortie::operator <<(const char* ob)
{
  if(bin_)
    {
      // Ca c'est dommage : dans LIST, il y a "<< blanc <<"
      // qui oblige a mettre ce test :
      // Il faudrait mettre "<< space <<"  mais cela change les fichiers
      // .Zones binaires...
      if (strcmp(ob, " "))
        {
          const int n = strlen(ob) + 1;
          ostream_->write((char *) ob, n * sizeof(char));
        }
    }
  else
    {
      (*ostream_) << ob;
    }
  // B.Mathieu, 7/10/2004 : je supprime l'espace apres ecrire(ob).
  // Attention, maintenant, il faut faire
  //  fichier << "chaine" << space << nombre;
  // pour pouvoir relire le fichier de facon transparente avec
  //  fichier >> motcle >> nombre;
  // Ancien code:
  //   if(!bin_)
  //     ecrire(" ");
  return *this;
}
#ifdef IO_avec_string
Sortie& Sortie::operator <<(const string& str)
{
  (*ostream_) << str;
  return (*this);
}
#endif
Sortie::~Sortie()
{
  if(ostream_)
    delete ostream_;
  ostream_=0;
}

// Description:
//  Change le mode d'ecriture du fichier.
//  Cette methode peut etre appelee n'importe quand. Attention
//  cependant pour les fichiers Ecrire_Fichier_Partage :
//  il faut faire le changement uniquement au debut de l'ecriture
//  d'un bloc, juste apres syncfile() (sinon, mauvaise traduction
//  des retours a la ligne lors du syncfile suivant).
int Sortie::set_bin(int bin)
{
  assert(bin==0 || bin==1);
  bin_ = bin;
  if (ostream_)
    {
      Cerr<<"Error you cant change binary format after open "<<finl;
      assert(0);
      Process::exit();
    }
  return bin_;
}
bool Sortie::has_ostream() const
{
  return (ostream_!=0);
}

int Sortie::is_bin()
{
  return bin_;
}

