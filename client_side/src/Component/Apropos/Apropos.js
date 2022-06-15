import * as React from 'react'
import nlp from '../../assets/nlp.png'
import "./Apropos.css";
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import AlternateEmailIcon from '@mui/icons-material/AlternateEmail';
import python from '../../assets/python.png'
import pytorch from '../../assets/pytorch.png'
import numpy from '../../assets/numpy.png'
import anaconda from '../../assets/anaconda.png'
import vscode from '../../assets/vscode.png'
import scikitlearn from '../../assets/scikitlearn.png'
const Apropos = () => {

  function Item(props) {
    const { sx, ...other } = props;
    return (
      <Box
        sx={{
          p: 1,
          m: 1,
          bgcolor: 'transparent',
          color: (theme) => (theme.palette.mode === 'dark' ? 'grey.300' : 'grey.800'),
          fontSize: '0.875rem',
          ...sx,
        }}
        {...other}
      />
    );
  }

  return (
    <div>
        <div id="nlp">
          <img src={nlp} height='90%' width='90%'></img>
        </div>
        <div id="apropos">
          <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#5885FB', marginTop: '3%', marginLeft: '3%' }}>
            À propos du projet
          </Typography>
          <hr/>
          <div id="description">
            <Typography sx={{ fontFamily: 'Poppins', fontSize: '18px', fontWeight: 300, color: '#00000', marginTop: '3%', marginLeft: '3%' }}>
            Notre projet de fin d'études, intitulé <b>“Évaluation de la cohérence du discours en utilisant les techniques d'apprentissage automatique”</b>, 
            vise à étudier et évaluer la cohérence du discours à plusieurs niveaux visant chacun des caractéristiques de cohérence bien définies (Sémantique : Parenté sémantique et information contextuelle, Syntaxique : Richesse lexicale).
            <br/>
            <br/>
            Pour ce, plusieurs modèles basés sur l'apprentissage profond (deep learning) sont proposés, conçus et implémentés. <b>La Coherencia</b> vous donne la main pour insérer votre propre texte ou série de documents, et sélectionner un modèle pour évaluer leurs cohérence selon le niveau d'analyse que vous souhaitez.
            <br/>Chaque document fourni en entrée se verra attribuer un score selon son degré de cohérence.
            </Typography>
          </div>
        </div>

        <div id="technologies">
          <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#5885FB', marginTop: '3%', marginLeft: '3%' }}>
            Technologies, librairies et outils utilisés
          </Typography>
          <hr/>
          <Box
                    sx={{
                    display: 'flex',
                    flexDirection: 'row',
                    p: 1,
                    m: 1,
                    justifyContent: 'space-between',
                    width: '80%'
                    }}
                >
                    <Item sx={{ backgroundColor: '#CADCF1', borderRadius: 2, width: '95px', height: '32px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={python} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Python</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFFAC9', borderRadius: 2, width: '125px', height: '32px' }}>
                        <div class='parent'>
                            <div class='child3'>
                                <img src={scikitlearn} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Scikit-learn</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#E1EFFF', borderRadius: 2, width: '95px', height: '32px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={numpy} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Numpy</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#E1EFFF', borderRadius: 2, width: '95px', height: '32px' }}>
                        <div class='parent'>
                            <div class='child'>
                                <img src={vscode} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>VS Code</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFF27B', borderRadius: 2, width: '125px', height: '32px' }}>
                        <div class='parent'>
                            <div class='child2'>
                                <img src={anaconda} class='logo'></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Anaconda</Typography>
                            </div>
                        </div>
                    </Item>
                    <Item sx={{ backgroundColor: '#FFE1E1', borderRadius: 2, width: '95px', height: '32px' }}>
                        <div class='parent'>
                            <div class='child'>                            
                                <img class='logo' src={pytorch}></img>
                                <Typography variant="caption" sx={{ fontFamily: 'Poppins', fontWeight: 500 }}>Pytorch</Typography>
                            </div>
                        </div>
                    </Item>
                </Box>
                    
        </div>
        
        <div id="realisedby">
          <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#5885FB', marginTop: '3%', marginLeft: '3%' }}>
            Réalisé par
          </Typography>
          <hr/>
          <div id="contact">
              <Box
              sx={{
                display: 'flex',
                flexDirection: 'row',
                bgcolor: 'transparent',
              }}
              >
                    <Box
                    sx={{
                      display: 'block',
                      p: 1,
                      m: 1,
                      bgcolor: 'transparent',
                      width: '50%'
                    }}
                    >
                      <Item>
                        <Typography sx={{ fontFamily: 'Poppins', fontSize: '18px', fontWeight: 300, color: '#00000', marginLeft: '3%' }}>
                          Lamia Medjahed
                        </Typography>
                        <Typography sx={{ fontFamily: 'Poppins', fontSize: '16px', fontWeight: 300, color: '#00000', marginLeft: '3%' }}>
                          Étudiante en 3CSSIL
                        </Typography>
                        <Typography sx={{ fontFamily: 'Didact Gothic', fontSize: '16px', fontWeight: 300, color: '#00000', marginLeft: '3%' }}>
                          <AlternateEmailIcon sx={{ color : '#5885FB', height: '5%', width: '5%' }} /> hl_medjahed@esi.dz
                        </Typography>
                      </Item>
                  </Box>
                  <Box
                    sx={{
                      display: 'block',
                      p: 1,
                      m: 1,
                      bgcolor: 'transparent',
                      width: '70%'
                    }}
                    >
                      <Item>
                        <Typography sx={{ fontFamily: 'Poppins', fontSize: '18px', fontWeight: 300, color: '#00000', marginLeft: '3%' }}>
                          Israa Hamdine
                        </Typography>
                        <Typography sx={{ fontFamily: 'Poppins', fontSize: '16px', fontWeight: 300, color: '#00000', marginLeft: '3%' }}>
                          Étudiante en 3CSSIL
                        </Typography>
                        <Typography sx={{ fontFamily: 'Didact Gothic', fontSize: '16px', fontWeight: 300, color: '#00000', marginLeft: '3%' }}>
                          <AlternateEmailIcon sx={{ color : '#5885FB', height: '3%', width: '3%' }} /> hi_hamdine@esi.dz
                        </Typography>
                      </Item>
                  </Box>
                  
                
              </Box>
          </div>
        </div>

        
        {/* <div id="encadrants">
          <Typography variant="h4" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#5885FB', marginTop: '3%', marginLeft: '3%' }}>
            encadré par
          </Typography>
          <hr/>
        </div> */}
        
    </div>
  )
}

export default Apropos