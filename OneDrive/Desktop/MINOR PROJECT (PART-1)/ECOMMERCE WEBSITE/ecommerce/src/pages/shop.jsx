import React from 'react'
import Hero from '../components/hero/hero'
import Popular from '../components/popular/popular'
import Offers from '../components/offers/offers'
import Newcollections from '../components/newcollections/newcollections'
import Newletters from '../components/newletters/newletters'
export default function Shop() {
    
  return (
    <div>
      <Hero></Hero>
      <Popular></Popular>
      <Offers></Offers>
      <Newcollections></Newcollections>
      <Newletters></Newletters>
    </div>
  )
}
