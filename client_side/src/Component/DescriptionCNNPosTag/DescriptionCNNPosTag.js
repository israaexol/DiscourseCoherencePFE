import React from 'react'
import Typography from '@mui/material/Typography';

const DescriptionCNNPosTag = () => {
  return (
    <div>
      <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
          <b>CNN_POS_TAG (Part Of Speech Tagging)</b> est un modèle neuronal à base de <b>CNN</b>
           (Convolutional Neural Networks) qui évalue la cohérence d'un discours 
          en analysant la nature et le séquencement de ses étiquettes morphosyntaxiques.
      </Typography>
      <br/>
      <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#079615' }}>
          Niveau d'analyse : Syntaxique
      </Typography>
    </div>
  )
}

export default DescriptionCNNPosTag