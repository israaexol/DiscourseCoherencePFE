import React, { useState, useEffect, useCallback } from 'react'
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import "./Sidebar.css";
import { experimentalStyled as styled } from '@mui/material/styles';
import Paper from '@mui/material/Paper';
import Link from '@mui/material/Link';

//import react pro sidebar components
import {
  ProSidebar,
  Menu,
  MenuItem,
  SidebarFooter,
  SidebarContent,
} from "react-pro-sidebar";

import github from '../../assets/github.png'

const Sidebar = ({ selectedIndex, descriptionList, accuracy, rappel, precision, f1_score, performances }) => {
  useEffect(() => {
    console.log(performances)
  }, []);

  // const [description, setDescriptionList] = useState(null)
  // const [performanceList, setPerformanceList] = useState(null)

  // const getDescription = () => {
  //   console.log(performances)
  //   setDescriptionList(descriptionList)
  // }

  // const getPerformances = () => {
  //   setPerformanceList(performances)
  // }

  function ModelDescription({ descriptionLists }) {


    return (
      <div>
        <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
          {/* <b>PARSEQ</b> est un modèle neuronal à base de <b>LSTM</b> (Long Short Term Memory)
                qui évalue la cohérence d'un discours à travers les similarités
                cosines entre <b>ses paragraphes</b> (i.e à un niveau plus global),
                en modélisant ainsi la parenté sémantique entre elles */}
          {descriptionLists ? descriptionLists[selectedIndex] : `Chargement...`}
        </Typography>
        <br />
        {/* <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 700, color: '#079615' }}>
            Niveau d'analyse : Sémantique
        </Typography> */}
      </div>)
  }

  return (
    <div id="header">
      <ProSidebar>
        <SidebarContent>
          <Card sx={{ width: 350, marginTop: '5%', marginLeft: '7%' }}>
            <CardContent>
              <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }} color="#5885FB" gutterBottom>
                Description du modèle
              </Typography>
              <ModelDescription descriptionLists={descriptionList} />
              {/* <ModelDescription/> */}
            </CardContent>
          </Card>

          <Card sx={{ width: 350, marginTop: '5%', marginLeft: '7%' }}>
            <CardContent>
              <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }} color="#5885FB" gutterBottom>
                Performances du modèle
              </Typography>
              <table>
                <tr>
                  <td><b>Exactitude</b></td>
                  <td>{accuracy ? accuracy[selectedIndex] : `...`}</td>
                  {/* <td>{performances ? performances[selectedIndex]["accuracy"] : `...`}</td> */}
                  {/* <ModelPerformances performance={performances} selected_index={selectedIndex}/> */}
                </tr>
                <tr>
                  <td><b>Précision</b></td>
                  <td>{precision ? precision[selectedIndex] : `...`}</td>
                  {/* <td>{performances ? performances[selectedIndex]["precision"] : `...`}</td> */}
                </tr>
                <tr>
                  <td><b>Rappel</b></td>
                  <td>{rappel ? rappel[selectedIndex] : `...`}</td>
                  {/* <td>{performances ? performances[selectedIndex]["rappel"] : `...`}</td> */}
                </tr>
                <tr>
                  <td><b>Score F1</b></td>
                  <td>{f1_score ? f1_score[selectedIndex] : `...`}</td>
                  {/* <td>{performances ? performances[selectedIndex]["F1_score"] : `...`}</td> */}
                </tr>

              </table>
            </CardContent>
          </Card>

          <Card sx={{ width: 350, marginTop: '5%', marginLeft: '7%' }}>
            <CardContent>
              <Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic' }} color="#5885FB" gutterBottom>
                Dataset utilisé
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'Poppins', fontWeight: 300 }}>
                <b>GCDC</b> <Link href="https://github.com/aylai/GCDC-corpus" sx={{ fontWeight: 500 }}>(Grammarly Corpus of Discourse Coherence)</Link>
                est un corpus introduit par <i>(Lai et Tetreault, 2018)</i> composé d'un total de 4800 documents provenant
                de 4 différents corpus, chacun relatif à un domaine (Emails d'<b>Enron</b>, Emails de <b>Clinton</b>, Réponses de forum <b>Yahoo</b> et Revues de business <b>Yelp</b>)

              </Typography>
            </CardContent>
          </Card>
        </SidebarContent>
        <SidebarFooter>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              p: 1,
              m: 1,
              justifyContent: 'flex-start',
              alignItems: 'flex-start'
            }}
          >
            <div class='parent'>
              <div class='child3'></div>
              <img src={github} height='7%' width='7%' class="logo3"></img>
              <Link target="_blank" href="https://github.com/israaexol/DiscourseCoherencePFE"><Typography sx={{ fontSize: 18, fontWeight: 'bold', fontFamily: 'Didact Gothic', marginLeft: '12px' }}>Répertoire GitHub</Typography></Link>
            </div>
            <div />
          </Box>
        </SidebarFooter>
      </ProSidebar>
    </div>
  )
}

export default Sidebar