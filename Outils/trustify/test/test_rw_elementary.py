"""
Testing loading of various **elementary pieces** of TRUST datasets into the datamodel generated by trustify.
Also test that the model can reproduce the initial dataset once it has been parsed.
"""

import os
import unittest

from test.reference_data import *
import trustify.base as base
import trustify.misc_utilities as mutil
from trustify.misc_utilities import ClassFactory, logger, TrustifyException
from trustify.trust_parser import TRUSTParser, TRUSTStream, TRUSTEndOfStreamException
import typing

########################################################################################
class TestCase(unittest.TestCase, mutil.UnitUtils):
    _test_dir = os.path.abspath(os.path.dirname(__file__))

    def import_and_gen_stream(self, simple_str, simplify):
        """ From a string, generates the corresponding token stream that can be passed to ReadFromTokens() methods """
        # Generate if needed
        self.generate_python_and_import("simple")
        self.mod = self._TRUG["simple"]

        # Parse the TRUST data set provided in arg
        if simplify:
            data_ex = mutil.simplify_successive_blanks(simple_str)
        else:
            data_ex = simple_str
        tp = TRUSTParser()
        tp.tokenize(data_ex)
        stream = TRUSTStream(tp)
        logger.debug("Token list (original)    : %s" % stream.tok)
        logger.debug("Token list (low stripped): %s" % stream.tokLow)
        logger.debug("Token list has %d items" % len(stream))
        return stream

    def builtin_test(self, builtin_cls, simple_str, simplify=True):
        """ Generic test method taking a builtin Python type, a string representing an extract of a data set
        and returning the corresponding stream, the value, and the parser object.
        @param builtin_cls a type as defined in Python 'typing' module, e.g. 'str', or 'List[float]'
        @param simple_st a string representing the piece of dataset to be parsed
        @param simplify whether to remove successive blanks etc. 
        """
        stream = self.import_and_gen_stream(simple_str, simplify)
        pars = base.Builtin_Parser.InstanciateFromBuiltin(builtin_cls)
        val = pars.readFromTokensBuiltin(stream)
        return stream, val, pars

    def generic_test(self, data, simplify=True):
        """ Generic test method taking a (piece of) dataset and testing it.
        This is a (much) simpler version of base.DataSet_Parser.ReadFromTokens()
        It returns the parser stream, and the object created.
        """
        stream = self.import_and_gen_stream(data, simplify)

        data_list = []  # A list representing the object found in the TRUST data file

        # Mimick (in a minimal fashion) what is done in DataSet.ReadFromTokens():
        cls_nam = stream.probeNextLow()
        ze_cls = ClassFactory.GetParserClassFromName(cls_nam)

        val = ze_cls.ReadFromTokens(stream)
        return stream, val

    #########################################################

    def test_utils(self):
        """ Test of type (from 'typing' module) manipulation methods """
        import typing as tp
        t1 = tp.Optional[tp.Annotated[tp.List[float], "tut"]]
        o, typs, ann = mutil.break_type(t1)
        self.assertTrue(o)
        self.assertEqual(typs, [list, float])
        self.assertEqual(ann, "tut")

        t1 = tp.Optional[tp.List[str]]
        o, typs, ann = mutil.break_type(t1)
        self.assertTrue(o)
        self.assertEqual(typs, [list, str])
        self.assertEqual(ann, None)

        t1 = tp.Annotated[tp.List[float], "tut"]
        o, typs, ann = mutil.break_type(t1)
        self.assertFalse(o)
        self.assertEqual(typs, [list, float])
        self.assertEqual(ann, "tut")

        t1 = tp.List[float]
        o, typs, ann = mutil.break_type(t1)
        self.assertFalse(o)
        self.assertEqual(typs, [list, float])
        self.assertEqual(ann, None)

        t1 = str
        o, typs, ann = mutil.break_type(t1)
        self.assertFalse(o)
        self.assertEqual(typs, [str])
        self.assertEqual(ann, None)

        t1 = tp.Optional[tp.Literal['flag1', 'flag2']]
        o, typs, ann = mutil.break_type(t1)
        self.assertTrue(o)
        self.assertEqual(typs, [tp.Literal, 'flag1', 'flag2'])
        self.assertEqual(ann, None)

        t1 = tp.Optional[str]
        o, typs, ann = mutil.break_type(t1)
        self.assertTrue(o)
        self.assertEqual(typs, [str])
        self.assertEqual(ann, None)

        t1 = tp.Optional[self.__class__]
        typ = mutil.strip_optional(t1)
        self.assertEqual(typ, self.__class__)

    def test_parser(self):
        """ Test of the TRUST parser """
        # Test parsing quotes
        data_ex = """ system "rm -rf */toto.lml.gz" one two"""
        tp = TRUSTParser()
        tp.tokenize(data_ex)
        stream = TRUSTStream(tp)
        self.assertEqual(stream.tokLow, ['', 'system', '"rm-rf*/toto.lml.gz"', 'one', 'two'])
        # Test EOF functionnality
        data_ex = """ # with comment #   toto      """
        stream, _, _ = self.builtin_test(str, data_ex, simplify=False)
        self.assertTrue(stream.eof())

    def test_builtin(self):
        """ Test parsing int and float """
        # Simple float first
        data_ex = """# comment #
    35.6"""
        stream, val, pars = self.builtin_test(float, data_ex, simplify=False)
        expec = 35.6
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())
        # Test writing out:
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)
        # Should fail:
        data_ex = "toto"
        # self.builtin_test(float, data_ex, simplify=False)
        self.assertRaisesRegex(TrustifyException, "Invalid float value: 'toto'", self.builtin_test, float, data_ex)

        # Simple int now:
        data_ex = """# comment #
    35"""
        stream, val, pars = self.builtin_test(int, data_ex, simplify=False)
        expec = 35
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())
        # Test writing out:
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)
        # Should fail:
        data_ex = "35.6"
        # self.builtin_test(int, data_ex, simplify=False)
        self.assertRaisesRegex(TrustifyException, "Invalid int value: '35.6'", self.builtin_test, int, data_ex)

        # Test parsing from typing.Literal type (corresponds to chaine(into=['single_val'] in the TRAD2)
        from typing import Literal
        data_ex = "# toto # single_val2"
        stream, val, pars = self.builtin_test(Literal["single_val1", "single_val2"], data_ex, simplify=False)
        expec = "single_val2"
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())
        # Test writing out:
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)
        # Should fail:
        data_ex = "toto"
        # self.builtin_test(int, data_ex, simplify=False)
        self.assertRaisesRegex(TrustifyException, "Invalid int value: 'toto'", self.builtin_test, int, data_ex)

    def test_simple_str(self):
        """ Test parsing simple string """
        data_ex = """
          # with many comments
            before
          #
          toto"""
        stream, val, pars = self.builtin_test(str, data_ex, simplify=False)
        expec = "toto"
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())
        # Test writing out
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)
        # Test inner braces in 'Chaine_Parser' (what makes 'bloc_lecture' work!)
        # In this specific situation (braces in the string), everything is kept in the resulting value, because we notably
        # want to remain case-sensitive.
        data_ex = """
          # with many comments
            before
          #
          { ta tu { toto } bouh }"""
        stream, val, pars = self.builtin_test(str, data_ex, simplify=False)
        expec = data_ex  # see comment above
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())
        # Test writing out
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)
        # Misformatted '{' should fail:
        data_ex = "} toto }"
        #self.builtin_test(str, data_ex, simplify=False)
        self.assertRaisesRegex(TrustifyException, "Misformatted string or block", self.builtin_test, str, data_ex)
        data_ex = "   {toto }"
        # self.builtin_test(str, data_ex, simplify=False)
        self.assertRaisesRegex(TrustifyException, "Misformatted string or block", self.builtin_test, str, data_ex)
        # When changing the value of a 'bloc lecture' we should always have a space to start with! See explanations in Chaine_Parser.toDatasetTokens()
        pars._pyd_value = "{ toto }"
        data_ex = " { toto }"
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)

    def test_builtin_list(self, fixed=False):
        """ Test parsing simple float list, with user defined size or fixed size """
        if fixed:
            sz_prefix = ""
            typ = typing.Annotated[typing.List[float], "size_is_dim"]
            # Pretend 'dimension' keyword has been read
            from trustify.base import Dimension_Parser
            Dimension_Parser._DIMENSION = 3
        else:
            sz_prefix = "3"
            typ = typing.List[float]

        data_ex = sz_prefix + " 48.5 89.2 18"
        stream, val, pars = self.builtin_test(typ, data_ex, simplify=False)
        expec = [48.5, 89.2, 18.0]
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())

        # Test writing out:
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)

        # Changing value on the model side should change output, and not reproduce initial tokens!!
        val[1] = 39.3
        new_s = sz_prefix + " 48.5 39.3 18"
        res = ''.join(pars.toDatasetTokens())
        self.assertEqual(res, new_s)
        val.pop()
        if not fixed:
            new_s = " 2 48.5 39.3"  # yes a space to start with, this is the default token when generated from scratch ...
            res = ''.join(pars.toDatasetTokens())
            self.assertEqual(res, new_s)
        else:
            # Changing size of a fixed size list should raise
            self.assertRaisesRegex(TrustifyException, "has a number of items which do not match problem dimension", pars.toDatasetTokens)

        # Same thing stressing parser:
        data_ex = sz_prefix + """ # yop #  48.5     \n    89.2 # comment inside #  18"""
        stream, val, pars = self.builtin_test(typ, data_ex, simplify=False)  # warning: no simplification here
        expec = [48.5, 89.2, 18.0]

        self.assertTrue(expec == val)
        self.assertTrue(stream.eof())
        s = ''.join(pars.toDatasetTokens())
        self.assertEqual(s, data_ex)

        # Changing value on the model side should change output, and not reproduce initial tokens!!
        val[0] = 39.3
        new_s = sz_prefix + " 39.3     \n    89.2 # comment inside #  18"""
        res = ''.join(pars.toDatasetTokens())
        self.assertEqual(res, new_s)
        if not fixed:
            val.pop()
            new_s =       " 2 39.3     \n    89.2""" # yes a space to start with, see comment above
            res = ''.join(pars.toDatasetTokens())
            self.assertEqual(res, new_s)

        # Ill formed lists
        data_ex = sz_prefix + " 48.5 89.2"
        # self.builtin_test(typing.List[float], data_ex)
        self.assertRaisesRegex(TRUSTEndOfStreamException, "Unexpected end of file",  self.builtin_test, typ, data_ex)
        data_ex = sz_prefix + " 48.5 89.2 sfsf"
        # self.builtin_test(typing.List[float], data_ex)
        self.assertRaisesRegex(TrustifyException, "Invalid float value: 'sfsf'", self.builtin_test, typ, data_ex)

        # Bad size:
        if not fixed:
            data_ex = "-24 48.5 89.2 18"
            # self.builtin_test(typ, data_ex, simplify=False)
            self.assertRaisesRegex(TrustifyException, "Invalid number of elements in list", self.builtin_test, typ, data_ex)

    def test_dim_list(self):
        """ Test list which size is fixed by the dimension of the problem """
        self.test_builtin_list(fixed=True)

    def test_curly_br(self):
        """ Test parsing and loading of a minimal TRUST dataset using curly braces """
        data_ex = """
        # Some stupid test #
        read_MED_bidon # with comments #
          {
             mesh  # coucou # ze_mesh_name
             file a/complicated/path/to.med
             exclude_groups 2 toto titi
             convertAllToPoly
          }"""
        for simplify in [True, False]:
            stream, res = self.generic_test(data_ex, simplify=simplify)
            exp = buildCurlyExpec(self.mod)
            self.assertEqual(exp, res)
            self.assertTrue(stream.eof())
            if not simplify:
                # Testing writing out
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Modifying data should change output!!
        res.convertalltopoly = False
        res.file = "new/file.med"
        s = ''.join(res.toDatasetTokens())
        new_s = """
        # Some stupid test #
        read_MED_bidon # with comments #
          {
             mesh  # coucou # ze_mesh_name
             file new/file.med
             exclude_groups 2 toto titi
          }"""
        self.assertTrue(mutil.check_str_equality(s, new_s).ok)

        # Ill-formed dataset - missing brace
        data_ex = """
        # Some stupid test #
        read_MED_bidon # with comments #

             mesh  # coucou # ze_mesh_name
          }"""
        # self.generic_test(data_ex)
        self.assertRaisesRegex(TrustifyException, "Keyword 'read_med_bidon|lire_med_bidon' expected a opening brace '{'", self.generic_test, data_ex)
        # Ill-formed dataset - missing brace
        data_ex = """
        # Some stupid test #
        read_MED_bidon # with comments # 
          {
             mesh  # coucou # ze_mesh_name
             convertAllToPoly"""
        self.assertRaisesRegex(TRUSTEndOfStreamException, "Unexpected end of file", self.generic_test, data_ex)

    def test_parser_stress(self):
        """ Basic test - stressing parser """
        data_ex = """read_MED_bidon { mesh ze_mesh_name file a/complicated/path/to.med exclude_groups 2 toto titi convertAllToPoly }"""
        stream, res = self.generic_test(data_ex, simplify=False)
        exp = buildCurlyExpec(self.mod)
        self.assertEqual(exp, res)
        self.assertTrue(stream.eof())
        # Test writing out:
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        data_ex += " toto"  # should still parse, 'toto' being the next keyword in the stream
        stream, res = self.generic_test(data_ex, simplify=False)
        self.assertEqual(exp, res)
        self.assertFalse(stream.eof())  # but not end of stream

    def test_list_complex(self):
        """ Test list of complex items """
        from typing import List, Annotated
        data_ex = """# a complex list # { 
        coucou {   }   ,
        coucou { graVITe Champ_Uniforme 2 0 9.8 }
        }"""
        # We need to be able to load Coucou class before calling builtin_test (to pass the correct type), so:
        self.generate_python_and_import("simple")
        self.mod = self._TRUG["simple"]
        C = self.mod.Coucou
        stream, val, pars = self.builtin_test(Annotated[List[C], "listcoucou"], data_ex, simplify=False)
        C, UF = self.mod.Coucou, self.mod.Uniform_field
        uf = UF()
        uf.val = [0, 9.8]
        expec = [C(), C()]
        expec[1].gravite = uf
        self.assertEqual(expec, val)
        self.assertTrue(stream.eof())

        # Test writing out:
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, data_ex).ok)

        # Changing value on the model side should change output, and not reproduce initial tokens!!
        val[0].gravite = UF()
        val[0].gravite.val = [28, 32]
        pars._pyd_value = val  # On a real dataset, this is done by _extendWithAttrTokens() in ConstrainBase_Parser
        expec = """# a complex list # { 
        coucou {
gravite  uniform_field 2 28.0 32.0   }   ,
        coucou { graVITe Champ_Uniforme 2 0 9.8 }
        }"""
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, expec).ok)

        # Removing an item from the list
        val.pop()
        pars._pyd_value = val
        expec = """# a complex list # { 
        coucou {
gravite  uniform_field 2 28.0 32.0   }
        }"""
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, expec).ok)

        # Adding an item to the list
        val.append(C())
        pars._pyd_value = val
        expec = """# a complex list # { 
        coucou {
gravite  uniform_field 2 28.0 32.0   }   , coucou {
 
}

        }"""  # Notice how the comma inside the list is output again as
              # was the initial one in the first list ...
        res = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(res, expec).ok)

        # Ill formed-list missing comma
        data_ex = """{  coucou {  } coucou {  }  }"""
        # self.builtin_test(typing.List[C], data_ex)
        self.assertRaisesRegex(TrustifyException, "expected a comma ", self.builtin_test, Annotated[List[C], "listcoucou"], data_ex)

        # Testing list of non-brace keywords:
        data_ex = """{ vx , vy }"""
        N = self.mod.Nom_anonyme
        stream, val, pars = self.builtin_test(Annotated[List[N], "list_nom_virgule"], data_ex, simplify=False)
        # Test writing out:
        s = ''.join(pars.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

    def test_opt_attr(self):
        """ Test attribute optionality. Missing required attribute should fail the parse. """
        data_ex = """
        # Invalid data with missing required attribute (file) #
        read_MED_bidon
          {
             mesh ze_mesh_name
             # file a/complicated/path/to.med #
             exclude_groups 2 toto titi
             convertAllToPoly
          }
        """
        # Should fail since mandatory attribue 'file' is missing:
        self.assertRaisesRegex(TrustifyException, "Attribute 'file' is mandatory for keyword 'read_med_bidon|lire_med_bidon'", self.generic_test, data_ex)

        # Keyword with all optionals should accept no attributes at all:
        data_ex = """coucou { }"""
        stream, res = self.generic_test(data_ex)
        MB = self.mod.Coucou
        exp = MB()
        self.assertEqual(exp, res)
        self.assertTrue(stream.eof())
        # Test writing out:
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

    def test_synonyms(self):
        """ Test synonyms """
        data_ex = """
        # Some stupid test with syno #
        read_MED_bidon
          {
             maillage ze_mesh_name
             fichier a/complicated/path/to.med
             exclude_groups 2 toto titi
             convertAllToPoly
          }"""
        for simplify in [True, False]:
            stream, res = self.generic_test(data_ex, simplify=simplify)
            exp = buildCurlyExpec(self.mod)
            self.assertEqual(exp, res)
            self.assertTrue(stream.eof())
            if not simplify:
                # Test writing out:
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

    def test_no_curly_br(self):
        """ Testing keywords with no curly braces """
        # Standard case
        data_ex = """uniform_field 3 34.6 12.8 90.9"""
        stream, res = self.generic_test(data_ex)
        UF = self.mod.Uniform_field
        # Build expected value
        exp = UF()
        exp.val = [34.6, 12.8, 90.9]
        self.assertEqual(exp, res)
        self.assertTrue(stream.eof())
        # Test writing out:
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # The below should parse - 'toto' can be considered the next keyword in the stream
        data_ex2 = data_ex + " toto "
        stream, res = self.generic_test(data_ex2)
        self.assertEqual(exp, res)
        #   here we re not at EOF, since 'toto' is there
        self.assertFalse(stream.eof())
        # With an optional flag and more spaces in between!
        data_ex = """
        # Example no curly brace with optional #
        uniform_field   flag_bidon  3   34.6  12.8 90.9
           flag_bidon2  2 15   29"""
        for simplify in [True, False]:
            stream, res = self.generic_test(data_ex, simplify=simplify)
            exp = UF()
            exp.val, exp.val2 = [34.6, 12.8, 90.9],  [15, 29]
            exp.flag_bidon, exp.flag_bidon2 = "flag_bidon", "flag_bidon2"
            self.assertEqual(exp, res)
            self.assertTrue(stream.eof())
            if not simplify:
                # Test writing out:
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Same story with next keyword:
        data_ex2 = data_ex + " toto "
        stream, res = self.generic_test(data_ex2)
        self.assertEqual(exp, res)
        self.assertFalse(stream.eof())

        # Ill-formed dataset (missing attribute):
        data_ex = """
        # Ill-formed #
        uniform_field flag_bidon
        """
        # self.generic_test(data_ex)
        self.assertRaisesRegex(TRUSTEndOfStreamException, "Unexpected end of file", self.generic_test, data_ex)
        # Ill-formed dataset (missing values)
        data_ex = """
        # Ill-formed #
        uniform_field 3 34.6 12.8
        """
        # self.generic_test(data_ex)
        self.assertRaisesRegex(TRUSTEndOfStreamException, "Unexpected end of file", self.generic_test, data_ex)

    def test_complex_attr(self):
        """ Testing complex attributes (with a prescribed type) """
        data_ex = """
          coucou {
            attr_bidon champ_uniforme 2 0  9.8  # using attr synonyms too #
          }"""
        stream, res = self.generic_test(data_ex, simplify=False)
        MB, UF = self.mod.Coucou, self.mod.Uniform_field
        # Build expected value
        exp, uf = MB(), UF()
        uf.val = [0, 9.8]
        exp.attr_bidon = uf
        self.assertEqual(exp, res)
        self.assertTrue(stream.eof())
        # Test writing out:
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Test changing one of the value
        res.attr_bidon.val[0] = -25
        data_ex = """
          coucou {
            attr_bidon champ_uniforme 2 -25  9.8  # using attr synonyms too #
          }"""
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Settng a complete new list
        res.attr_bidon.val = [5,6,7]  # Complete new list
        data_ex = """
          coucou {
            attr_bidon champ_uniforme 3 5.0 6.0 7.0  # using attr synonyms too #
          }""" # Note how the comment is attached to the closing brace...
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Seting an invalid type (pydantic should complain):
        import pydantic
        self.assertRaisesRegex(pydantic.ValidationError, "validation error for Uniform_field", setattr, res.attr_bidon, "val", "qsd")

        # Disabling an attribute
        res.attr_bidon = None
        data_ex = """
          coucou {  # using attr synonyms too #
          }"""
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Assigning a complete new object for an attribute
        res.gravite = UF()
        res.gravite.val = [2,3]
        data_ex = """
          coucou {
gravite  uniform_field 2 2.0 3.0  # using attr synonyms too #
          }""" # I know indentation sucks ...
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Adding unexpected attribute should raise:
        data_ex = """ coucou { blop }"""
        # self.generic_test(data_ex, simplify=False)
        self.assertRaisesRegex(TrustifyException, "Unexpected attribute 'blop' in keyword 'coucou'", self.generic_test, data_ex)

    def test_add_rm_attr(self):
        """ Testing adding an attribute to a keyword and checking output ok """
        data_ex = """
        # Some stupid test with syno #
        read_MED_bidon
          {
             maillage ze_mesh_name
             fichier a/complicated/path/to.med
             exclude_groups 2 toto titi
             convertAllToPoly
          }"""

        stream, res = self.generic_test(data_ex, simplify=False)
        exp = buildCurlyExpec(self.mod)
        self.assertEqual(exp, res)
        self.assertTrue(stream.eof())
        # Adding a (valid) attribute
        res.no_family_names_from_group_names = True
        data_expected = """
        # Some stupid test with syno #
        read_MED_bidon
          {
             maillage ze_mesh_name
             fichier a/complicated/path/to.med
             exclude_groups 2 toto titi
             convertAllToPoly
no_family_names_from_group_names 
          }"""
        # (Yes I know this is poorly indented and space after "no_fam..." needs to be kept!)
        # Test writing out:
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_expected).ok)
        # Removing an attribute (setting to None):
        res.mesh = None
        data_expected = """
        # Some stupid test with syno #
        read_MED_bidon
          {
             fichier a/complicated/path/to.med
             exclude_groups 2 toto titi
             convertAllToPoly
no_family_names_from_group_names 
          }"""
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_expected).ok)

    def test_inheritance(self):
        """ Testing inheritance - gravite expects a 'field_base' of which 'champ_uniform' is a child
        Also test inherited members: bidon should inherit member 'top_attr' from toto_base
        """
        data_ex = """
          coucou {
            gravite champ_uniforme 2 0 9.8
          }"""
        for simplify in [True, False]:
            stream, res = self.generic_test(data_ex, simplify=simplify)
            MB, UF = self.mod.Coucou, self.mod.Uniform_field
            # Build expected value
            exp, uf = MB(), UF()
            uf.val = [0, 9.8]
            exp.gravite = uf
            # Test
            self.assertEqual(exp, res)
            self.assertTrue(stream.eof())
            if not simplify:
                # Test writing out:
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Changing inherited attribute should change output:
        FB = self.mod.Field_base
        res.gravite = FB()
        s = ''.join(res.toDatasetTokens())
        new_s = """
          coucou {
            gravite field_base
          }"""
        self.assertTrue(mutil.check_str_equality(s, new_s).ok)

        # Wrong inheritance should raise:
        data_ex = """
          coucou {
            gravite coucou { }
          }
        """
        # offset, res = self.generic_test(data_ex)
        self.assertRaisesRegex(Exception, "Invalid TRUST keyword: 'coucou'", self.generic_test, data_ex)

        # Member inheritance
        data_ex = """bidon { 
            top_attr_syno nimp
            ma_liste { }
            }"""
        stream, res = self.generic_test(data_ex, simplify=False)
        exp = self.mod.Bidon()
        exp.top_attr = "nimp"
        exp.ma_liste = []
        self.assertEqual(exp, res)
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

    def test_forward_decl_1(self):
        """ Testing forward declaration and 'read' keyword.
        This uses the main loop from Dataset. """
        from trustify.base import Declaration_Parser
        # Simple forward declaration parsing
        data_ex = """  # with comments #
        lire_med_bidon     rmed"""
        for simplify in [True, False]:
            stream = self.import_and_gen_stream(data_ex, simplify)
            exp = self.mod.Declaration()
            exp.identifier, exp.ze_type = "rmed", self.mod.Read_med_bidon
            res = Declaration_Parser.ReadFromTokens(stream)
            self.assertTrue(stream.eof())
            self.assertEqual(exp, res)
            if not simplify:
                # Test writing out:
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)
        # Test invalid decl
        data_ex = """ blop blip """
        stream = self.import_and_gen_stream(data_ex, False)
        self.assertRaisesRegex(TrustifyException, "Invalid TRUST keyword: 'blop'",  Declaration_Parser.ReadFromTokens, stream)

    def test_forward_decl_2(self):
        """ Forward declaration test. With a non-interprete keyword. """
        # Valid dataset
        data_ex = """champ_uniforme gravite
                     # comment in the middle #
                     read    gravite  2   0.0    9.8"""
        stream = self.import_and_gen_stream(data_ex, simplify=False)
        ds_cls = self.mod.Dataset_Parser
        res = ds_cls.ReadFromTokens(stream)
        self.assertEqual(len(res.entries), 2)
        self.assertTrue(stream.eof())
        UF, D, RD = self.mod.Uniform_field, self.mod.Declaration, self.mod.Read
        decl = D()
        decl.identifier, decl.ze_type = "gravite", UF
        r, f, l = RD(), UF(), [0,9.8]
        f.val = l
        r.identifier = "gravite"
        r.obj = f
        self.assertEqual(res.entries[0], decl)
        self.assertEqual(res.entries[1], r)
        # Test writing out:
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Changing read object completely:
        f2 = UF()
        f2.val = [5,6,7]
        res.entries[1].obj = f2
        s = ''.join(res.toDatasetTokens())
        data_ex = """champ_uniforme gravite
                     # comment in the middle #
                     read    gravite 3 5.0 6.0 7.0"""
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

        # Invalid datasets
        #    Misformatted fwd decl:
        data_ex = """champ_uniforme 35
                     coucou { }"""
        stream = self.import_and_gen_stream(data_ex, simplify=False)
        # res = ds_cls.ReadFromTokens(stream)
        self.assertRaisesRegex(TrustifyException, "Invalid identifier '35' in forward declaration of type 'champ_uniforme'", ds_cls.ReadFromTokens, stream)
        #    Misformatted - trying to make a forward declaration with an interprete:
        data_ex = """champ_uniforme gravite
                     lire_med forward"""
        stream = self.import_and_gen_stream(data_ex, simplify=False)
        # res = ds_cls.ReadFromTokens(stream)
        self.assertRaisesRegex(TrustifyException, "Keyword 'read_med|lire_med' expected a opening brace ", ds_cls.ReadFromTokens, stream)
        #    Wrong reference:
        data_ex = """champ_uniforme gravite
                     read toto"""
        stream = self.import_and_gen_stream(data_ex, simplify=False)
        # res = ds_cls.ReadFromTokens(stream)
        self.assertRaisesRegex(TrustifyException, "Referencing object 'toto' .* not declared", ds_cls.ReadFromTokens, stream)

    def test_forward_decl_3(self):
        from trustify.base import Dataset_Parser
        # Small datasets using forward declarations
        data_ex = """
        lire_med_bidon   rmed    # bidon because not an interpret #
        coucou cb

        Read  rmed
          {
             mesh ze_mesh_name
             file a/complicated/path/to.med
          }

        read    cb {   }"""

        for simplify in [True, False]:
            stream = self.import_and_gen_stream(data_ex, simplify)
            res = Dataset_Parser.ReadFromTokens(stream)
            self.assertEqual(len(res.entries), 4)
            self.assertTrue(stream.eof())
            exp0, exp1, exp2, exp3 = buildForwardExpec(self.mod)
            self.assertEqual(exp0, res.entries[0]) # Forward decl
            self.assertEqual(exp1, res.entries[1]) # Coucou cb
            self.assertEqual(exp2, res.entries[2]) # read rmed
            self.assertEqual(exp3, res.entries[3]) # read cb
            if not simplify:
                # Test writing out:
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

    def test_dim_pars(self):
        """ Test Dimension_Parser class - output was buggy """
        from trustify.base import Dataset_Parser
        data_ex = "dimension # coucou # 2"
        stream = self.import_and_gen_stream(data_ex, simplify=False)
        res = Dataset_Parser.ReadFromTokens(stream)
        s = ''.join(res.toDatasetTokens())
        self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

    def test_complete_dataset(self):
        """ Complete (small) dataset with foward declarations etc... """
        from trustify.base import Dataset_Parser
        data_ex = """
        # Some stupid test #
        champ_uniforme gravite
        nom coucou
        read gravite 2 28 32   # Keyword with no brace #
        read coucou toto
        lire_MED
          {
             mesh ze_mesh_name
             file a/complicated/path/to.med
             exclude_groups 2 toto titi
             convertAllToPoly
          }"""

        for simplify in [True, False]:
            stream = self.import_and_gen_stream(data_ex, simplify)
            res = Dataset_Parser.ReadFromTokens(stream)
            self.assertEqual(len(res.entries), 5)
            self.assertTrue(stream.eof())

            exp = buildMinimalExpec(self.mod)
            for i in range(5):
                self.assertEqual(exp[i], res.entries[i])
            if not simplify:
                # Test writing out:
                s = ''.join(res.toDatasetTokens())
                self.assertTrue(mutil.check_str_equality(s, data_ex).ok)

if __name__ == '__main__':
    unittest.main()
